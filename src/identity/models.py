"""Identity 数据模型：定义人设结构。"""

from pydantic import BaseModel, Field


class TriggerRule(BaseModel):
    """触发规则：决定哪些场景下激活此人设。"""

    groups: list[str] = Field(default_factory=list, description="适用群号列表，空则不限")
    keywords: list[str] = Field(default_factory=list, description="触发关键词，消息包含则激活")


class Identity(BaseModel):
    """一个人设。"""

    id: str = Field(description="唯一标识，如 'default', 'catgirl', 'butler'")
    name: str = Field(description="人设名称，如 '猫娘', '管家'")
    personality: str = Field(description="核心性格描述，会写入 System Prompt")
    trigger: TriggerRule = Field(default_factory=TriggerRule)
    priority: int = Field(default=0, description="优先级，数字越大优先级越高")
