"""Compatibility tests for textual DeepSeek/DSML tool calls."""

from src.llm.client import _extract_dsml_tool_calls


def test_extracts_multiple_dsml_tool_calls() -> None:
    text = """<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="web_search">
<｜｜DSML｜｜parameter name="max_results" string="false">8</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="query" string="true">梗 2026年 流行 网络用语</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
<｜｜DSML｜｜invoke name="get_hot_trends">
<｜｜DSML｜｜parameter name="limit" string="false">10</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>"""

    cleaned, calls = _extract_dsml_tool_calls(text)

    assert cleaned == ""
    assert len(calls) == 2
    assert calls[0].name == "web_search"
    assert calls[0].input == {"max_results": 8, "query": "梗 2026年 流行 网络用语"}
    assert calls[1].name == "get_hot_trends"
    assert calls[1].input == {"limit": 10}


def test_preserves_natural_text_outside_dsml_block() -> None:
    text = "我先查一下。<||DSML||tool_calls><||DSML||invoke name=\"web_search\">"
    text += "<||DSML||parameter name=\"query\" string=\"true\">测试梗</||DSML||parameter>"
    text += "</||DSML||invoke></||DSML||tool_calls>"

    cleaned, calls = _extract_dsml_tool_calls(text)

    assert cleaned == "我先查一下。"
    assert calls[0].input["query"] == "测试梗"


def test_drops_malformed_dsml_instead_of_leaking_it() -> None:
    cleaned, calls = _extract_dsml_tool_calls("<｜｜DSML｜｜tool_calls>残缺内容")

    assert cleaned == ""
    assert calls == []
