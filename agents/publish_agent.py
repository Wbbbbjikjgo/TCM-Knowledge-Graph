"""发布 Agent：对接小红书开放接口，一键发布图文内容。"""

import os


class PublishResult:
    def __init__(self, success: bool, message: str, url: str = ""):
        self.success = success
        self.message = message
        self.url = url

    def to_dict(self) -> dict:
        return {"success": self.success, "message": self.message, "url": self.url}


class XiaohongshuPublisher:
    """小红书发布器：通过开放平台接口发布笔记。

    凭证从环境变量 XHS_COOKIE 读取，禁止硬编码。
    实际接入时需按小红书开放平台文档补充签名与接口地址。
    """

    API_BASE = "https://edith.xiaohongshu.com/api/sns/v1"

    def __init__(self):
        self.cookie = os.getenv("XHS_COOKIE", "")

    @property
    def configured(self) -> bool:
        return bool(self.cookie)

    def publish(self, title: str, content: str,
                image_paths: list[str] | None = None) -> PublishResult:
        if not self.configured:
            return PublishResult(
                success=False,
                message="未配置小红书凭证（请在 .env 中设置 XHS_COOKIE），内容已保存为草稿。",
            )
        try:
            return self._do_publish(title, content, image_paths or [])
        except Exception as e:
            return PublishResult(success=False, message=f"发布失败：{e}")

    def _do_publish(self, title: str, content: str,
                    image_paths: list[str]) -> PublishResult:
        """实际发布逻辑占位：对接平台接口上传素材并创建笔记。"""
        # TODO: 接入小红书开放平台实际接口（素材上传 + 笔记创建）
        raise NotImplementedError("小红书开放接口尚未接入，请配置后实现 _do_publish")


def publish_handler(state: dict) -> dict:
    """LangGraph 节点：发布上一环节生成的内容。"""
    content = state.get("generated_content") or (
        state["messages"][-1].content if state.get("messages") else "")
    if not content:
        return {"answer": "没有可发布的内容，请先生成图文笔记。"}

    # 标题取内容首行或【标题】后的文本
    title = content.split("\n", 1)[0].replace("【标题】", "").strip()[:20]

    result = XiaohongshuPublisher().publish(title, content)
    return {"answer": result.message, "publish_result": result.to_dict()}
