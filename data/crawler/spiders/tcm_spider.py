"""通用中医药文献采集 Spider（可动态指定链接与域名，通用正文选择器）。"""

from datetime import datetime

import scrapy

from data.crawler.items import TcmDocItem

# 通用正文选择器（按优先级回退，覆盖常见文章站结构）
_CONTENT_SELECTORS = [
    "div.content p::text",
    "article p::text",
    "div.article p::text",
    "div.article-content p::text",
    "div#content p::text",
]


class TcmSpider(scrapy.Spider):
    """通用爬虫：抓取给定 URL 的标题与正文。

    两种用法：
    1. 动态链接（前端/脚本调用）：实例化时传入 start_urls / allowed_domains，
       默认只抓取给定页面本身（follow_links=False）。
    2. 命令行骨架：scrapy crawl tcm，如需在列表页递归跟踪详情与翻页，
       设置 follow_links=True 并按站点调整 parse 中的选择器。
    """

    name = "tcm"
    start_urls = ["https://example.com/tcm/articles"]
    allowed_domains = ["example.com"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,  # 礼貌爬取，避免对目标站点造成压力
        "ROBOTSTXT_OBEY": True,
    }

    def __init__(self, start_urls=None, allowed_domains=None, follow_links=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if start_urls:
            self.start_urls = list(start_urls)
        if allowed_domains:
            self.allowed_domains = list(allowed_domains)
        self.follow_links = follow_links

    def start_requests(self):
        # 前端传入的链接通常是文章详情页，直接抽取正文
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse_detail, errback=self._on_error)

    def _on_error(self, failure):
        self.logger.warning("请求失败 %s: %s", failure.request.url, failure.value)

    def parse(self, response):
        # 列表页（仅 follow_links=True 时）：跟踪详情链接 + 翻页
        for href in response.css("a.article-link::attr(href)").getall():
            yield response.follow(href, callback=self.parse_detail)
        next_page = response.css("a.next-page::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_detail(self, response):
        item = self._extract_item(response)
        if item["content"]:
            yield item

    def _extract_item(self, response) -> TcmDocItem:
        item = TcmDocItem()
        item["source"] = response.url
        item["title"] = self._extract_title(response)
        item["section"] = ""
        item["content"] = self._extract_content(response)
        item["crawl_time"] = datetime.now().isoformat()
        return item

    def _extract_title(self, response) -> str:
        for selector in ("h1::text", "title::text", "h2::text"):
            title = response.css(selector).get()
            if title and title.strip():
                return title.strip()
        return ""

    def _extract_content(self, response) -> str:
        content = "\n".join(
            p.strip() for p in response.css(", ".join(_CONTENT_SELECTORS)).getall()
        ).strip()
        if not content:
            content = "\n".join(
                p.strip() for p in response.css("p::text").getall()
            ).strip()
        return content
