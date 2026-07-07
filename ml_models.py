"""여러 머신러닝 모델로 정적 심볼을 분류해 인식 성능을 비교.

모든 모델은 동일한 25차원 feature vector(features.extract)를 입력으로 사용한다.
보정 데이터(심볼당 30 샘플)는 한 번만 수집하면 되고, 각 모델은 그 데이터를
재사용해 학습한다 — 모델마다 다시 동작을 입력할 필요 없음.

표본이 심볼당 30개뿐이라 대형 신경망(딥러닝)은 과적합 위험이 커서 제외하고,
classical ML(SVM, RandomForest, LogisticRegression) + 작은 MLP 하나를 비교군으로 둔다.
"""

import warnings

import numpy as np

from sign_classifier import classify_calibrated_vec

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _build_sklearn_models() -> dict:
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=12, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "MLP": MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=0),
    }


class ModelManager:
    """심볼 모드(letters/digits)별로 여러 모델을 보정 데이터에 학습해 비교 제공."""

    def __init__(self):
        # {"letters": {model_name: fitted_model}, "digits": {...}}
        self._fitted: dict[str, dict] = {"letters": {}, "digits": {}}
        # k-NN(custom)은 학습이 아니라 거리 비교라 원본 보정 데이터를 그대로 보관
        self._cal: dict[str, dict] = {"letters": {}, "digits": {}}

    def fit(self, cal_data: dict | None) -> None:
        """보정 데이터를 letters/digits로 나눠 등록된 모든 모델을 학습."""
        self._fitted = {"letters": {}, "digits": {}}
        self._cal = {"letters": {}, "digits": {}}
        if not cal_data:
            return

        self._cal["letters"] = {k: v for k, v in cal_data.items() if not k.isdigit()}
        self._cal["digits"] = {k: v for k, v in cal_data.items() if k.isdigit()}

        if not _SKLEARN_AVAILABLE:
            print("[ml_models] scikit-learn 미설치 — kNN(custom)만 사용합니다.")
            return

        for subset_name, subset in self._cal.items():
            if len(subset) < 2:
                continue   # 클래스가 2개 미만이면 분류기 학습 불가
            X = np.concatenate(list(subset.values()), axis=0)
            y = np.concatenate([[label] * len(samples) for label, samples in subset.items()])
            for name, model in _build_sklearn_models().items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X, y)
                    self._fitted[subset_name][name] = model
                except Exception as e:
                    print(f"[ml_models] {name} 학습 실패 ({subset_name}): {e}")

    def model_names(self) -> list[str]:
        """현재 사용 가능한 모델 이름 목록 (kNN(custom) 포함)."""
        names = ["kNN (custom)"]
        for fitted in self._fitted.values():
            for name in fitted:
                if name not in names:
                    names.append(name)
        return names

    def predict(self, vec: np.ndarray, is_digit: bool) -> dict[str, str]:
        """이미 추출된 feature vector에 대해 모델별 예측 결과 dict 반환."""
        subset_name = "digits" if is_digit else "letters"
        cal_subset = self._cal.get(subset_name, {})

        results: dict[str, str] = {}
        results["kNN (custom)"] = classify_calibrated_vec(vec, cal_subset) if cal_subset else "?"

        for name, model in self._fitted.get(subset_name, {}).items():
            try:
                pred = model.predict(vec.reshape(1, -1))[0]
                results[name] = str(pred)
            except Exception:
                results[name] = "?"
        return results
