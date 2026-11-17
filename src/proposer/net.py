"""URL validation, the SSRF guard and reachability checks.

Everything the assistant fetches or checks goes through here. Hosts that
resolve to private, loopback or link-local addresses are refused on every
hop, unless PROPOSER_ALLOW_PRIVATE_URLS is set (tests, closed networks).
"""
import asyncio
import ftplib
import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx

USER_AGENT = 'proposer'
URL_SCHEMES = ('http', 'https', 'ftp')
URL_TIMEOUT = 10          # seconds, per URL check


def private_urls_allowed() -> bool:
    return os.environ.get('PROPOSER_ALLOW_PRIVATE_URLS', '').lower() in ('1', 'true', 'yes')


def check_addresses(host: str, addresses):
    """Refuse hosts that resolve to private, loopback or link-local addresses."""
    if private_urls_allowed():
        return
    for address in addresses:
        ip = ipaddress.ip_address(address.split('%')[0])
        if getattr(ip, 'ipv4_mapped', None):
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise ValueError('%s resolves to a private address' % host)


def resolve_host(host: str, port: int):
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValueError('host not found: %s' % host)
    return [info[4][0] for info in infos]


def validate_url(url: str) -> str:
    """The URL if it is http, https or ftp with a host, else ValueError."""
    if not isinstance(url, str) or any(c.isspace() for c in url.strip()):
        raise ValueError('Not a valid URL: %s' % url)
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in URL_SCHEMES:
        raise ValueError('Only http, https and ftp URLs are supported: %s' % url)
    if not parsed.hostname:
        raise ValueError('The URL has no host: %s' % url)
    return url


def public_url(url: str) -> str:
    """validate_url, and refuse hosts on private networks (SSRF)."""
    url = validate_url(url)
    parsed = urlparse(url)
    port = parsed.port or {'http': 80, 'https': 443, 'ftp': 21}[parsed.scheme]
    check_addresses(parsed.hostname, resolve_host(parsed.hostname, port))
    return url


def status_text(status: int) -> str:
    reasons = {401: 'access denied', 403: 'access denied', 404: 'not found',
               410: 'not found', 429: 'too many requests'}
    return '%s (%d)' % (reasons.get(status, 'server error' if status >= 500
                                    else 'request failed'), status)


async def guard_request(request: httpx.Request):
    """httpx hook, run for each request including redirects."""
    if private_urls_allowed():
        return
    host = request.url.host
    port = request.url.port or (443 if request.url.scheme == 'https' else 80)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValueError('host not found: %s' % host)
    check_addresses(host, [info[4][0] for info in infos])


def _ftp_size(url: str) -> dict:
    parsed = urlparse(url)
    check_addresses(parsed.hostname, resolve_host(parsed.hostname, parsed.port or 21))
    try:
        with ftplib.FTP(timeout=URL_TIMEOUT) as ftp:
            ftp.connect(parsed.hostname, parsed.port or 21)
            ftp.login(parsed.username or 'anonymous', parsed.password or '')
            ftp.voidcmd('TYPE I')
            size = ftp.size(parsed.path)
        return {'ok': True, 'status': None, 'size': size, 'error': None}
    except ftplib.error_perm as error:
        message = 'not found' if str(error).startswith('550') else str(error)
        return {'ok': False, 'status': None, 'size': None, 'error': message}


def _size_from(response: httpx.Response) -> int | None:
    content_range = response.headers.get('Content-Range', '')
    if '/' in content_range and content_range.rsplit('/', 1)[1].isdigit():
        return int(content_range.rsplit('/', 1)[1])
    length = response.headers.get('Content-Length')
    # Content-Length of a compressed or ranged response is not the file size.
    if length and length.isdigit() and response.status_code == 200 \
            and not response.headers.get('Content-Encoding'):
        return int(length)
    return None


async def _check_http(client: httpx.AsyncClient, url: str) -> dict:
    response = await client.head(url)
    if response.status_code in (403, 405, 501):
        # HEAD not allowed: ask for the first byte instead.
        async with client.stream('GET', url, headers={'Range': 'bytes=0-0'}) as response:
            pass
    status = response.status_code
    if status >= 400:
        return {'ok': False, 'status': status, 'size': None, 'error': status_text(status)}
    return {'ok': True, 'status': status, 'size': _size_from(response), 'error': None}


async def check_url(client: httpx.AsyncClient, url: str) -> dict:
    """One check: {url, ok, status, size, error}."""
    result = {'url': url}
    try:
        url = validate_url(url)
        if urlparse(url).scheme == 'ftp':
            result.update(await asyncio.wait_for(asyncio.to_thread(_ftp_size, url),
                                                 URL_TIMEOUT * 2))
        else:
            result.update(await _check_http(client, url))
    except ValueError as error:
        result.update(ok=False, status=None, size=None, error=str(error))
    except (httpx.TimeoutException, TimeoutError):
        result.update(ok=False, status=None, size=None, error='timed out')
    except httpx.TooManyRedirects:
        result.update(ok=False, status=None, size=None, error='too many redirects')
    except (httpx.HTTPError, OSError, ftplib.Error) as error:
        result.update(ok=False, status=None, size=None,
                      error='could not connect (%s)' % (str(error) or type(error).__name__))
    return result


async def check_urls(urls: list[str]) -> list[dict]:
    """Check each URL with a headers-only request; results in the same order."""
    limits = asyncio.Semaphore(16)
    async with httpx.AsyncClient(
            follow_redirects=True, timeout=URL_TIMEOUT, max_redirects=10,
            headers={'User-Agent': USER_AGENT},
            event_hooks={'request': [guard_request]}) as client:
        async def one(url):
            async with limits:
                return await check_url(client, url)
        return await asyncio.gather(*(one(url) for url in urls))
