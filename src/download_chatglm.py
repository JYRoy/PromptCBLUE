#模型下载
from modelscope import snapshot_download

repo_id = "ZhipuAI/ChatGLM-6B"
downloaded = snapshot_download(
    repo_id,
)