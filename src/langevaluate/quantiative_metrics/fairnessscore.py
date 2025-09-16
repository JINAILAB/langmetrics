import numpy as np
from typing import Sequence, Optional, Union

ArrayLike = Union[Sequence[int], Sequence[float], Sequence[str], np.ndarray]

class FairnessScore:
    """
    공정성 점수 (0~1, 높을수록 공정)
    - metric: min(group_mean) / max(group_mean)
    - type:
        * 'sex' : groups가 0/1, 숫자 ID, 문자열 라벨 등 카테고리인 경우
        * 'age' : groups가 연령(예: 22, 33, 45); 자동으로 10년 단위 bin ('10-20','20-30',...)
    - scores: 각 샘플의 성능 점수(0~1 권장; clip으로 0~1 범위 보장)
    """
    def __init__(self, bin_width: int = 10, min_samples_per_group: int = 1):
        self.bin_width = int(bin_width)
        self.min_samples_per_group = int(min_samples_per_group)
        self.last_stats = None  # 최근 호출의 요약 통계

    @staticmethod
    def _ensure_1d(a) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1:
            raise ValueError("Input must be 1D or (N,1) shaped.")
        return a

    def _bin_ages(self, ages: ArrayLike) -> np.ndarray:
        a = self._ensure_1d(ages).astype(float)
        if np.any(np.isnan(a)):
            raise ValueError("ages contain NaN.")
        if self.bin_width <= 0:
            raise ValueError("bin_width must be positive.")
        # 10년 단위 등급: [start, end) 구간 라벨 'start-end'
        starts = (np.floor(a / self.bin_width) * self.bin_width).astype(int)
        ends = starts + self.bin_width
        labels = np.array([f"{s:d}-{e:d}" for s, e in zip(starts, ends)], dtype=object)
        return labels

    def _groups_from_type(self, groups: ArrayLike, type: str) -> np.ndarray:
        t = (type or "sex").lower()
        if t not in ("sex", "age"):
            raise ValueError("type must be 'sex' or 'age'.")
        if t == "sex":
            g = self._ensure_1d(groups)
            return g  # 숫자/문자 라벨 모두 허용
        else:  # 'age'
            return self._bin_ages(groups)

    def __call__(
        self,
        groups: ArrayLike,
        scores: ArrayLike,
        type: str = "sex",
        sample_weight: Optional[ArrayLike] = None,
    ) -> float:
        g = self._groups_from_type(groups, type=type)
        s = self._ensure_1d(scores).astype(float)
        if s.shape[0] != g.shape[0]:
            raise ValueError("groups and scores must have the same length.")

        # 가중치 처리
        if sample_weight is None:
            w = np.ones_like(s, dtype=float)
        else:
            w = self._ensure_1d(sample_weight).astype(float)
            if w.shape[0] != s.shape[0]:
                raise ValueError("sample_weight length must match scores.")

        # 점수는 0~1로 보정
        s = np.clip(s, 0.0, 1.0)

        # 그룹별 가중 평균
        uniq = np.unique(g)
        means = []
        by_group = {}
        for grp in uniq:
            mask = (g == grp)
            if np.sum(mask) < self.min_samples_per_group:
                continue
            denom = np.sum(w[mask])
            if denom <= 0:
                continue
            m = float(np.average(s[mask], weights=w[mask]))
            means.append(m)
            by_group[str(grp)] = m

        # 유효 그룹이 1개 이하 → 편차 없다고 간주
        if len(means) <= 1:
            self.last_stats = {"by_group": by_group, "gap": 0.0, "min": None, "max": None}
            return 1.0

        max_m = float(np.max(means))
        min_m = float(np.min(means))
        fairness = 1.0 if max_m == 0.0 else float(min_m / max_m)
        fairness = float(np.clip(fairness, 0.0, 1.0))

        self.last_stats = {"by_group": by_group, "gap": max_m - min_m, "min": min_m, "max": max_m}
        return fairness