from pathlib import Path
from pydantic import BaseModel, AnyUrl, FilePath, DirectoryPath, Field, field_validator
from typing import List, Optional, Any
import os

class VideoLink(BaseModel):
    links: List[AnyUrl]

class VideoFileStatus(BaseModel):
    filename: str
    filepath: FilePath
    codec: Optional[str] = None
    needs_conversion: Optional[bool] = None
    duration: Optional[float] = None

class TrackNetCommand(BaseModel):
    video_path: FilePath
    save_dir: DirectoryPath
    gpu_id: int
    thread_id: int

    @field_validator("save_dir", mode="before")
    @classmethod
    def ensure_directory_path(cls, v):
        os.makedirs(v, exist_ok=True)
        return Path(v)
        
class Img2Court(BaseModel):
    video_path: FilePath
    H: List[List[float]]

    @field_validator("H", mode="before")
    @classmethod
    def ensure_2d_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("H must be a 2D matrice.")
        for row in v:
            if not isinstance(row, list):
                raise ValueError("Each row in H must be a list floats.")
            for item in row:
                if not isinstance(item, (int, float)):
                    raise ValueError("All elements in H must be floats.")
        return v
   
class PlayerPosition(BaseModel):
    c_x: float = Field(ge = -1, lt = 6.5)
    c_y: float = Field(ge = -1, lt = 14.5)
    x1: float
    y1: float
    x2: float
    y2: float
    
class SegmentsFile(BaseModel):
    segments: List[List[int]]

class PlayerTracksFile(BaseModel):
    match_id: str
    rally: List[Any]

class ShuttleTracksFile(BaseModel):
    match_id: str
    frames: List[Any]


