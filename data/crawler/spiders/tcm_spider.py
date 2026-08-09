"""通用中医药文献采集 Spider（按目标站点定制选择器）。"""

from datetime import datetime

import scrapy

from data.crawler.items import TcmDocItem


class TcmSpider(scrapy.Spider):
    """定向爬虫骨架：将 start_urls 指向目标中医药文献站点，
    并按站点 DOM 结构调整 parse 中的 CSS 选择器。
    """

    name = "tcm"
    # 示例入口，部署时替换为实际数据源
    start_urls = ["https://example.com/tcm/articles"]
    allowed_domains = ["example.com"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,  # 礼貌爬取，避免对目标站点造成压力
        "ROBOTSTXT_OBEY": True,
    }

    def parse(self, response):
        # 列表页：跟踪详情链接
        for href in response.css("a.article-link::attr(href)").getall():
            yield response.follow(href, callback=self.parse_detail)
        # 翻页
        next_page = response.css("a.next-page::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_detail(self, response):
        item = TcmDocItem()
        item["source"] = response.url
        item["title"] = response.css("h1::text").get(default="").strip()
        item["section"] = ""
        item["content"] = "\n".join(
            p.strip() for p in response.css("div.content p::text").getall()
        ).strip()
        item["crawl_time"] = datetime.now().isoformat()
        if item["content"]:
            yield item
