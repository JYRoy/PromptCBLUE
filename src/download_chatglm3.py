#模型下载
from modelscope import snapshot_download

repo_id = "ZhipuAI/chatglm3-6b"
downloaded = snapshot_download(
    repo_id,
)