import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio
import requests
from tqdm import tqdm

from llamantin.collector import Collector
from llamantin.config import settings
from llamantin.docsearchagent import DocSearchAgent
from llamantin.llm import LLMProvider


def download_file(url, session, save_path):
    response = session.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(save_path, "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def download_and_cache_data(data_dir: str):
    base_url = "https://api.github.com/repos/Unstructured-IO/unstructured/contents/example-docs"
    params = {"ref": "b092fb7f474cc585d14db6773f25a0b1c62f2e82"}
    download_path = os.path.join(data_dir, "example-docs")

    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        files = response.json()

        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for file in files:
                    if file["type"] == "file":
                        file_url = file["download_url"]
                        save_path = os.path.join(download_path, file["name"])
                        futures.append(
                            executor.submit(download_file, file_url, session, save_path)
                        )

                for future in tqdm(futures, total=len(futures), unit="file"):
                    future.result()

    return download_path


@pytest_asyncio.fixture
async def doc_agent():
    settings.MODEL_TEMPERATURE = 0.0
    data_dir = "test_data"
    data_path = download_and_cache_data(data_dir)
    collector = Collector(data_path)
    await collector.initialize_database()
    llm = LLMProvider.create_llm(settings=settings)
    return DocSearchAgent(llm, settings, collector, 0.25)


@pytest.mark.asyncio
async def test_docsearchagent_basic_search(doc_agent):
    query = "Describe the performance evalutation a standing trustee must undergo"
    result = await doc_agent.search(query)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Reduced Compensation" in result
