from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class STINGCell:
   
    level: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    count: int = 0
    temp_mean: float = np.nan
    temp_max: float = np.nan
    temp_min: float = np.nan
    temp_std: float = np.nan

    cluster_id: int = -1
    is_relevant: bool = False       
    confidence: float = 0.0        

    children: List["STINGCell"] = field(default_factory=list)

    @property
    def lat_center(self) -> float:
        return (self.lat_min + self.lat_max) / 2

    @property
    def lon_center(self) -> float:
        return (self.lon_min + self.lon_max) / 2

    @property
    def lat_size(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def lon_size(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def area_deg2(self) -> float:
        return self.lat_size * self.lon_size

    def is_populated(self) -> bool:
        return self.count > 0

    def temperature_range(self) -> float:
        if np.isnan(self.temp_max) or np.isnan(self.temp_min):
            return np.nan
        return self.temp_max - self.temp_min

    def contains_point(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "lat_center": self.lat_center,
            "lon_center": self.lon_center,
            "count": self.count,
            "temp_mean": round(self.temp_mean, 3) if not np.isnan(self.temp_mean) else None,
            "temp_max": round(self.temp_max, 3) if not np.isnan(self.temp_max) else None,
            "temp_min": round(self.temp_min, 3) if not np.isnan(self.temp_min) else None,
            "temp_std": round(self.temp_std, 3) if not np.isnan(self.temp_std) else None,
            "cluster_id": self.cluster_id,
            "is_relevant": self.is_relevant,
            "confidence": round(self.confidence, 4),
        }

    def __repr__(self):
        return (f"STINGCell(L{self.level} "
                f"lat=[{self.lat_min:.1f},{self.lat_max:.1f}] "
                f"lon=[{self.lon_min:.1f},{self.lon_max:.1f}] "
                f"n={self.count} T={self.temp_mean:.1f}°C "
                f"cluster={self.cluster_id})")