import os
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from neuralforecast.auto import AutoMLP
from neuralforecast import NeuralForecast
from logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"
logger = get_logger(task_name="model")


class Forecaster:
    """A wrapper class for the MLP Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "MLP Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        local_scaler_type: str = None,
        num_samples: int = 10,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new MLP Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.



            local_scaler_type (str): Scaler to apply per-serie to all features before fitting, which is inverted after predicting.
                Can be 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'

            num_samples (int): Number of trails to use for the AutoMLP model.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.data_schema = data_schema
        self.local_scaler_type = local_scaler_type
        self.num_samples = num_samples
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs
        self.history_length = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        config_automl = lambda trial: {
            "max_steps": 1000,
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1e-4, 1e-3, 1e-2]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "hidden_size": trial.suggest_categorical("hidden_size", [25, 50, 100, 200]),
            "num_lr_decays": 3,
            "input_size": trial.suggest_categorical(
                "input_size",
                [
                    data_schema.forecast_length * 3,
                    data_schema.forecast_length * 4,
                    data_schema.forecast_length * 5,
                ],
            ),
            "random_seed": self.random_state,
        }

        self.config_automl = config_automl

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by neuralforecast.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """

        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency == ["secondly"]:
            return "S"
        else:
            return 1

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the training data by converting the index to datetime if available
        and drops or keeps other covariates depending on use_exogenous.

            Args:
                data (pd.DataFrame): The training data.
        """

        if self.data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            data[self.data_schema.time_col] = pd.to_datetime(
                data[self.data_schema.time_col]
            )

        groups_by_ids = data.groupby(self.data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())

        all_series = [groups_by_ids.get_group(id_).reset_index() for id_ in all_ids]

        if self.history_length:
            for index, series in enumerate(all_series):
                all_series[index] = series.iloc[-self.history_length :]
            data = pd.concat(all_series).drop(columns="index")

        if self.data_schema.future_covariates:
            data.drop(columns=self.data_schema.future_covariates, inplace=True)

        if self.data_schema.static_covariates:
            data.drop(columns=self.data_schema.static_covariates, inplace=True)

        if self.data_schema.past_covariates:
            data.drop(columns=self.data_schema.past_covariates, inplace=True)

        data.rename(
            columns={
                self.data_schema.id_col: "unique_id",
                self.data_schema.time_col: "ds",
                self.data_schema.target: "y",
            },
            inplace=True,
        )

        return data

    def generate_static_exogenous(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the dataframe of static covariates

        Args:
            history (pd.DataFrame): The prepared dataframe of history.

        Returns (pd.DataFrame): The static covariates dataframe expected by neuralforecast.
        """
        static_exogenous = history.groupby("unique_id").first().reset_index()
        static_exogenous = static_exogenous[
            ["unique_id"] + self.data_schema.static_covariates
        ]
        return static_exogenous

    def generate_future_exogenous_for_predict(
        self, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        futr_df = test_data[["unique_id", "ds"] + self.data_schema.future_covariates]

        if self.data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            futr_df["ds"] = pd.to_datetime(futr_df["ds"])

        return futr_df

    def _validate_lags_and_history_length(self, series_length: int):
        """
        Validate the value of lags and that history length is at least double the forecast horizon.
        If the provided lags value is invalid (too large), lags are set to the largest possible value.

        Args:
            series_length (int): The length of the history.

        Returns: None
        """
        forecast_length = self.data_schema.forecast_length
        if series_length < 2 * forecast_length:
            raise ValueError(
                f"Training series is too short. History should be at least double the forecast horizon. history_length = ({series_length}), forecast horizon = ({forecast_length})"
            )

        if self.lags >= series_length:
            self.lags = series_length - 1
            logger.warning(
                f"The provided lags value >= available history length. Lags are set to to (history length - 1) = {series_length-1}"
            )

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.

        Args:
            history (pandas.DataFrame): The features of the training data.

        """
        np.random.seed(self.random_state)

        history = self.prepare_data(history)

        models = [
            AutoMLP(
                h=self.data_schema.forecast_length,
                num_samples=self.num_samples,
                config=self.config_automl,
                backend="optuna",
            )
        ]

        self.model = NeuralForecast(
            models=models,
            freq=self.map_frequency(self.data_schema.frequency),
            local_scaler_type=self.local_scaler_type,
        )

        self.model.fit(df=history)
        self._is_trained = True
        self.history = history

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The prediction dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        test_data.rename(
            columns={
                self.data_schema.id_col: "unique_id",
                self.data_schema.time_col: "ds",
                self.data_schema.target: "y",
            },
            inplace=True,
        )

        forecast = self.model.predict(df=self.history)

        forecast[prediction_col_name] = forecast.drop(columns=["ds"]).mean(axis=1)
        forecast.reset_index(inplace=True)
        forecast["ds"] = test_data["ds"]
        forecast.rename(
            columns={
                "unique_id": self.data_schema.id_col,
                "ds": self.data_schema.time_col,
            },
            inplace=True,
        )
        return forecast

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        self.model.save(model_dir_path, save_dataset=False, overwrite=True)
        del self.model
        del self.config_automl
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model_object = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        model = NeuralForecast.load(model_dir_path)
        model_object.model = model
        return model_object

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
