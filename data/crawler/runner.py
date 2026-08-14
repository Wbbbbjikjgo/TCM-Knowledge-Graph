"""爬虫运行器：命令行抓取入口 + 面向前端的 crawl_urls API。

命令行用法（独立进程，避免 Twisted reactor 单例无法重复启动的问题）：
    python -m data.crawler.runner <url> [<url> ...] -o <out.json>
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from common.config import PROJECT_ROOT

USER_AGENT = "TCM-Knowledge-Graph/1.0 (educational crawler)"


def _spider_for(urls: list[str]):
    """构造针对给定 URL 的爬虫类（动态 start_urls / allowed_domains）。"""
    from data.crawler.spiders.tcm_spider import TcmSpider

    domains = sorted({urlparse(u).netloc for u in urls if urlparse(u).netloc})

    class FrontendSpider(TcmSpider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_urls = list(urls)
            if domains:
                self.allowed_domains = domains

    return FrontendSpider


def _run_scrapy(urls: list[str], out_path: str) -> None:
    """在本进程内运行 Scrapy，把抓取到的 item 导出到 out_path（JSON 数组）。"""
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(
        settings={
            "FEEDS": {out_path: {"format": "json", "encoding": "utf-8", "overwrite": True}},
            "ROBOTSTXT_OBEY": True,
            "DOWNLOAD_DELAY": 1.0,
            "USER_AGENT": USER_AGENT,
            "LOG_LEVEL": "WARNING",
            "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
        }
    )
    process.crawl(_spider_for(urls))
    process.start()


def _check_url(url: str) -> tuple[str, str]:
    """返回 (状态, 说明)：状态为 allowed / invalid / robots_disallowed。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "invalid", "链接无效（缺少 http/https 协议或域名）"

    rp = RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        # robots.txt 不可达时不拦截，交由 Scrapy 处理
        return "allowed", ""
    if not rp.can_fetch(USER_AGENT, url):
        return "robots_disallowed", "目标站点 robots.txt 禁止抓取该链接"
    return "allowed", ""


def crawl_urls(urls: list[str]) -> dict:
    """抓取多个 URL，返回 {"items": [...], "blocked": [...], "error": str|None}。

    - items：抓取成功且含正文的文档列表（source/title/section/content/crawl_time）。
    - blocked：不允许抓取或无效的链接及原因（robots.txt 禁止 / 链接无效）。
    - error：抓取过程异常信息（如 scrapy 未安装、网络错误等）。
    """
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        return {"items": [], "blocked": [], "error": "未提供任何链接"}

    allowed, blocked = [], []
    for u in urls:
        status, reason = _check_url(u)
        if status == "allowed":
            allowed.append(u)
        else:
            blocked.append({"url": u, "reason": reason})

    items: list[dict] = []
    error = None

    if allowed:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "crawl.json")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "data.crawler.runner",
                     *allowed, "-o", out_path],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    error = (proc.stderr or proc.stdout or "未知错误").strip()[-500:]
                elif Path(out_path).exists():
                    with open(out_path, encoding="utf-8") as f:
                        items = json.load(f)
            except Exception as e:  # noqa: BLE001
                error = str(e)

    return {"items": items, "blocked": blocked, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 TcmSpider 抓取指定链接")
    parser.add_argument("urls", nargs="+", help="目标链接（可多个）")
    parser.add_argument("-o", "--out", required=True, help="结果 JSON 输出路径")
    args = parser.parse_args()
    _run_scrapy(args.urls, args.out)


if __name__ == "__main__":
    main()
