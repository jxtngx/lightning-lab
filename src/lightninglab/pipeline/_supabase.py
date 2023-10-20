# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import List, Union

import supabase
from dotenv import load_dotenv
from supabase import create_client as _create_client

load_dotenv()


class Supabase:
    """Wrapper for Supabase-Py

    Args:
        url: The Supabase project URL
        key: The Supabase project key
    """

    def __init__(self, url: str, key: str) -> None:
        self.url: str = url
        self.key: str = key

    def create_client(self) -> None:
        """start the Supabase Client session"""
        _create_client(self.url, self.key)

    def create_bucket(self, bucketname: str) -> None:
        """creates a Supabase bucket

        Args:
            bucketname: name of source Supabase bucket
        """
        supabase.storage.create_bucket(bucketname)

    def list_buckets(self) -> List:
        """lists all buckets available to the user"""
        return supabase.storage.list_buckets()

    def list_bucket_files(self, bucketname: str) -> List:
        """lists all files in a given bucket

        Args:
            bucketname: name of source Supabase bucket
        """
        return supabase.storage.from_(bucketname).list()

    def download_file(self, bucketname: str, source: str, destination: Union[Path, str]) -> None:
        """downloads a file from a given bucket

        Args:
            bucketname: name of source Supabase bucket
            source: name of file to download
            destination: path to download file to
        """
        with open(destination, "wb+") as f:
            res = supabase.storage.from_(bucketname).download(source)
            f.write(res)

    def upload_file(self, bucketname: str, source: Union[Path, str], destination: str, file_options: dict):
        """uploads a file from a given bucket

        Args:
            bucketname: name of source Supabase bucket
            source: path of file to upload
            destination: bucket location to upload to
            file_options: see https://supabase.com/docs/reference/javascript/storage-from-upload
        """
        with open(source, "rb") as f:
            supabase.storage.from_(bucketname).upload(
                file=f,
                path=destination,
                file_options=file_options,
            )
