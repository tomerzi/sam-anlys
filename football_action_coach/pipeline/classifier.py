import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import xgboost as xgb
from config import Config


def rubric_label_name(label: int, classes_rubric: np.ndarray) -> str:
    cr = {int(x) for x in np.asarray(classes_rubric).flatten()}
    if cr == {0, 1}:
        return "BAD" if int(label) == 0 else "GOOD"
    if cr == {0, 2}:
        return "BAD" if int(label) == 0 else "GOOD"
    names = {0: "BAD", 1: "OK", 2: "GOOD"}
    return names.get(int(label), f"CLASS_{label}")


def _rubric_to_encoded(y_rubric: np.ndarray, classes_order: np.ndarray) -> np.ndarray:
    mapping = {int(c): i for i, c in enumerate(classes_order)}
    return np.array([mapping[int(v)] for v in y_rubric], dtype=int)


class ActionClassifier:
    """
    XGBoost classifier for action quality (binary or multi-class on rubric 0/1/2).
    Leave-One-Out CV across videos.
    """

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.model: xgb.XGBClassifier | None = None
        self.classes_rubric_: np.ndarray | None = None

    def _build_xgb(self, y_train_enc: np.ndarray, n_classes: int) -> xgb.XGBClassifier:
        if n_classes == 2:
            n0 = (y_train_enc == 0).sum()
            n1 = (y_train_enc == 1).sum()
            spw = float(n0) / (float(n1) + 1e-8)
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                scale_pos_weight=spw,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        return xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

    def leave_one_out_cv(self, video_datasets: list[dict]) -> dict:
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs_pos: list[float] = []

        n_folds = len(video_datasets)
        global_classes = np.sort(
            np.unique(np.concatenate([np.asarray(v["y"], dtype=int) for v in video_datasets]))
        )

        for i, test_set in enumerate(video_datasets):
            train_sets = [v for j, v in enumerate(video_datasets) if j != i]

            X_train = np.concatenate([v["X"] for v in train_sets])
            y_train = np.concatenate([v["y"] for v in train_sets]).astype(int)
            X_test = test_set["X"]
            y_test = np.asarray(test_set["y"], dtype=int)

            classes_train = np.sort(np.unique(y_train))
            mask = np.isin(y_test, classes_train)
            if not mask.all():
                dropped = int((~mask).sum())
                print(f"  [Warn] Fold test={test_set['video_id']}: dropped {dropped} windows with labels unseen in train.")
            X_test, y_test = X_test[mask], y_test[mask]
            if len(y_test) == 0:
                print(f"  [Warn] Fold test={test_set['video_id']}: no eval windows left; skip fold.")
                continue

            y_tr_enc = _rubric_to_encoded(y_train, classes_train)
            y_te_enc = _rubric_to_encoded(y_test, classes_train)
            n_classes = len(classes_train)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            clf = self._build_xgb(y_tr_enc, n_classes)
            clf.fit(X_tr_s, y_tr_enc, eval_set=[(X_te_s, y_te_enc)], verbose=False)

            pred_enc = clf.predict(X_te_s).astype(int)
            proba = clf.predict_proba(X_te_s)
            preds_rubric = classes_train[pred_enc]

            all_preds.extend(preds_rubric.tolist())
            all_labels.extend(y_test.tolist())

            if n_classes == 2:
                all_probs_pos.extend(proba[:, 1].tolist())
            else:
                for row in range(proba.shape[0]):
                    j = int(np.argmax(proba[row]))
                    all_probs_pos.append(float(proba[row, j]))

            names = [rubric_label_name(int(c), classes_train) for c in classes_train]
            print(f"\n── Fold {i+1}/{n_folds}  test={test_set['video_id']} ──")
            print(
                classification_report(
                    y_test,
                    preds_rubric,
                    labels=classes_train.tolist(),
                    target_names=names,
                    zero_division=0,
                )
            )

        y_true = np.asarray(all_labels, dtype=int)
        y_hat = np.asarray(all_preds, dtype=int)
        overall_names = [rubric_label_name(int(c), global_classes) for c in global_classes]
        print("\n══ Overall LOO-CV ══")
        print(
            classification_report(
                y_true,
                y_hat,
                labels=global_classes.tolist(),
                target_names=overall_names,
                zero_division=0,
            )
        )
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true, y_hat, labels=global_classes.tolist()))

        f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
        auc = float("nan")
        if len(global_classes) == 2:
            y_bin = (y_true == int(global_classes.max())).astype(int)
            try:
                auc = float(roc_auc_score(y_bin, np.asarray(all_probs_pos)))
            except ValueError:
                auc = float("nan")

        print(f"Macro-F1: {f1:.3f}")
        if len(global_classes) == 2 and not np.isnan(auc):
            print(f"AUC (positive class = {int(global_classes.max())}): {auc:.3f}")

        return {"auc": auc, "f1": f1, "preds": all_preds, "labels": all_labels}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ActionClassifier":
        y = np.asarray(y, dtype=int)
        self.classes_rubric_ = np.sort(np.unique(y))
        y_enc = _rubric_to_encoded(y, self.classes_rubric_)
        n_classes = len(self.classes_rubric_)
        X_s = self.scaler.fit_transform(X)
        self.model = self._build_xgb(y_enc, n_classes)
        self.model.fit(X_s, y_enc, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.classes_rubric_ is None:
            raise RuntimeError("Model not fitted or loaded.")
        X_s = self.scaler.transform(X)
        proba = self.model.predict_proba(X_s)
        pred_enc = np.argmax(proba, axis=1)
        labels = self.classes_rubric_[pred_enc]
        proba_max = proba.max(axis=1)
        return labels.astype(int), proba_max.astype(np.float64), proba.astype(np.float64)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "xgb_model.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.classes_rubric_, path / "classes_rubric.pkl")
        print(f"[Classifier] Model saved → {path}")

    def load(self, path: Path) -> "ActionClassifier":
        self.model = joblib.load(path / "xgb_model.pkl")
        self.scaler = joblib.load(path / "scaler.pkl")
        cr_path = path / "classes_rubric.pkl"
        if cr_path.exists():
            self.classes_rubric_ = joblib.load(cr_path)
        else:
            self.classes_rubric_ = np.array([0, 1], dtype=int)
        return self
