from datetime import datetime
from typing import override, Literal

from requests import get, post, patch, Response

from mipcandy.frontend.prototype import Frontend
from mipcandy.types import Settings


class NotionFrontend(Frontend):
    def __init__(self, secrets: Settings) -> None:
        super().__init__(secrets)
        self._api_key: str = self.require_nonempty_secret("notion_api_key", require_type=str)
        self._database_id: str = self.require_nonempty_secret("notion_database_id", require_type=str)
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        self._num_epochs: int = 1
        self._early_stop_tolerance: int = -1
        self._start_time: str = ""
        self._page_id: str = ""

    def retrieve_database(self) -> Response:
        return get(f"https://api.notion.com/v1/databases/{self._database_id}", headers=self._headers)

    def query_database(self, *, experiment_id: str | None = None) -> Response:
        json = {"filter": {"property": "Experiment ID", "title": {"equals": experiment_id}}} if experiment_id else None
        return post(f"https://api.notion.com/v1/databases/{self._database_id}/query", json=json, headers=self._headers)

    def select_experiment(self, experiment_id: str) -> str:
        experiments = self.query_database(experiment_id=experiment_id)
        if experiments.status_code != 200:
            raise RuntimeError(f"Failed to query database: {experiments.json()}")
        experiments = experiments.json()["results"]
        if len(experiments) == 1:
            return experiments[0]["id"]
        if len(experiments) > 1:
            raise RuntimeError(f"Found multiple experiments with the same ID {experiment_id}")
        return ""

    def new_experiment(self, experiment_id: str, trainer: str, model: str, note: str, num_macs: float,
                       num_params: float) -> Response:
        self._start_time = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S.000%z")
        properties = {
            "Experiment ID": {"title": [{"text": {"content": experiment_id}}]},
            "Status": {"status": {"name": "In Progress"}},
            "Progress": {"number": 0},
            "Early Stop": {"number": 1},
            "Trainer": {"select": {"name": trainer}},
            "Model": {"select": {"name": model}},
            "Time": {"date": {"start": self._start_time}},
            "Note": {"rich_text": [{"text": {"content": note}}]},
            "MACs (G)": {"number": round(num_macs, 1)},
            "Params (M)": {"number": round(num_params, 1)},
            "Epoch": {"number": 0},
            "Score": {"number": 0},
        }
        page_id = self.select_experiment(experiment_id)
        if page_id:
            self._page_id = page_id
            return patch(f"https://api.notion.com/v1/pages/{page_id}", json={"properties": properties},
                         headers=self._headers)
        res = post("https://api.notion.com/v1/pages", json={
            "parent": {"database_id": self._database_id},
            "icon": {"external": {"url": "https://www.notion.so/icons/science_gray.svg"}},
            "properties": properties
        }, headers=self._headers)
        self._page_id = res.json()["id"]
        return res

    def update_experiment(self, experiment_id: str, status: Literal["In Progress", "Completed", "Interrupted"],
                          *, epoch: int | None = None, score: float | None = None,
                          early_stop_tolerance: int | None = None, observation: str | None = None) -> Response:
        if not self._page_id:
            raise RuntimeError(f"Experiment {experiment_id} has not been created")
        properties = {"Status": {"status": {"name": status}}}
        if epoch is not None:
            properties["Progress"] = {"number": epoch / self._num_epochs}
            properties["Epoch"] = {"number": epoch}
        if early_stop_tolerance is not None:
            properties["Early Stop"] = {"number": max(early_stop_tolerance, 0) / self._early_stop_tolerance}
        if score is not None:
            properties["Score"] = {"number": round(score, 4)}
        if observation is not None:
            properties["Observation"] = {"rich_text": [{"text": {"content": observation}}]}
        if status == "Completed":
            properties["Progress"] = {"number": 1}
            properties["Time"] = {"date": {"start": self._start_time,
                                           "end": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S.000%z")}}
        return patch(f"https://api.notion.com/v1/pages/{self._page_id}", json={"properties": properties},
                     headers=self._headers)

    @override
    def on_experiment_created(self, experiment_id: str, trainer: str, model: str, note: str, num_macs: float,
                              num_params: float, num_epochs: int, early_stop_tolerance: int) -> None:
        self._num_epochs = num_epochs
        self._early_stop_tolerance = early_stop_tolerance
        res = self.new_experiment(experiment_id, trainer, model, note, num_macs * 1e-9, num_params * 1e-6)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to create experiment: {res.json()}")

    @override
    def on_experiment_updated(self, experiment_id: str, epoch: int, metrics: dict[str, list[float]],
                              early_stop_tolerance: int) -> None:
        try:
            self.update_experiment(experiment_id, "In Progress", epoch=epoch, score=max(metrics["val score"]),
                                   early_stop_tolerance=early_stop_tolerance)
        except RuntimeError:
            pass

    @override
    def on_experiment_completed(self, experiment_id: str) -> None:
        res = self.update_experiment(experiment_id, "Completed")
        if res.status_code != 200:
            raise RuntimeError(f"Failed to update experiment: {res.json()}")

    @override
    def on_experiment_interrupted(self, experiment_id: str, error: Exception) -> None:
        res = self.update_experiment(experiment_id, "Interrupted", observation=repr(error))
        if res.status_code != 200:
            raise RuntimeError(f"Failed to update experiment: {res.json()}")
