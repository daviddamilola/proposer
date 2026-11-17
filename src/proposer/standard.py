"""The standard read tools every pack gets: papers, folders, tables, URL checks."""
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from . import read
from .pack import Pack, Tool

S = {'type': 'string'}


class ReadPaperArgs(BaseModel):
    doi_or_url: str = Field(min_length=3)


class ListDirectoryArgs(BaseModel):
    url: str = Field(min_length=6)
    pattern: str | None = None


class ReadTableArgs(BaseModel):
    url: str = Field(min_length=6)
    sheet: str | None = None
    start_row: int = Field(0, ge=0)
    max_rows: int = Field(read.TABLE_ROWS, ge=1)


class CheckUrlsArgs(BaseModel):
    urls: list[str] = Field(min_length=1)


def _host(url):
    return urlparse(url).hostname or url


def _parameters(properties, required):
    return {'type': 'object', 'properties': properties, 'required': required}


def tools(pack: Pack) -> list[Tool]:
    async def read_paper(args, turn):
        return await read.read_paper(args.doi_or_url, pack.paper)

    async def list_directory(args, turn):
        return await read.list_directory(args.url, args.pattern)

    async def read_table(args, turn):
        return await read.read_table(args.url, args.sheet, args.start_row, args.max_rows)

    async def check_urls(args, turn):
        return await read.check_urls_tool(args.urls)

    standard = [
        Tool('read_paper', 'Read a paper by DOI or URL: title, data availability, methods '
             'excerpts, supplementary files with URLs, inline tables, links.',
             ReadPaperArgs, read_paper, status='Reading the paper…',
             parameters=_parameters({'doi_or_url': S}, ['doi_or_url'])),
        Tool('list_directory', 'List a folder (HTTP index page or FTP): file names and sizes.',
             ListDirectoryArgs, list_directory,
             status=lambda a: 'Listing %s…' % _host(a.url),
             parameters=_parameters(
                 {'url': S, 'pattern': {'type': 'string',
                                        'description': 'Optional glob, e.g. *.csv.gz'}},
                 ['url'])),
        Tool('read_table', 'Read a CSV, TSV or XLSX table (also a member of a zip: url#member).',
             ReadTableArgs, read_table,
             status=lambda a: 'Reading %s…' % (
                 a.url.split('#')[-1].rsplit('/', 1)[-1] or 'the table'),
             parameters=_parameters(
                 {'url': S, 'sheet': S, 'start_row': {'type': 'integer'},
                  'max_rows': {'type': 'integer'}}, ['url'])),
        Tool('check_urls', 'Check whether URLs can be downloaded (headers only).',
             CheckUrlsArgs, check_urls,
             status=lambda a: 'Checking %d links…' % len(a.urls),
             parameters=_parameters({'urls': {'type': 'array', 'items': S}}, ['urls'])),
    ]
    return [tool for tool in standard if tool.name not in pack.without]
