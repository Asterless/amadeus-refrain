"""Identity 数据模型：定义人设结构。"""

from pydantic import BaseModel, Field


class Identity(BaseModel):
    """人设。"""

    id: str = Field(default="default", description="标识")
    name: str = Field(description="人设名称")
    personality: str = Field(description="核心性格描述，会写入 System Prompt")
    proactive: str | None = Field(default=None, description="主动插话判断规则，None 表示不主动插话")
