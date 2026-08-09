"""爬虫数据项定义。"""

import scrapy


class TcmDocItem(scrapy.Item):
    """中医药文献原始文本数据项。"""

    source = scrapy.Field()      # 来源站点/URL
    title = scrapy.Field()       # 文档标题
    section = scrapy.Field()     # 章节
    content = scrapy.Field()     # 正文文本
    crawl_time = scrapy.Field()  # 采集时间
