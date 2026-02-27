from typing import Union, Literal, Annotated, ClassVar
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    NonNegativeInt
)
import lightgbm as lgb


class CBParams(BaseModel):
    """CatBoost model training parameters. For details, see
    [CatBoost's documentation](
    https://catboost.ai/docs/en/references/training-parameters/).

    Args:
        objective (Literal): Loss function in training. Defaults to Logloss.
        eval_metric (Literal): Metric to compare models. Defaults to Accuracy.
        auto_class_weights (Literal): Class weights.
        depth (int): Limit the max depth for tree model.
        l2_leaf_reg (float): Coefficient of the L2 regularization term.
        learning_rate (float): Learning rate.
        rsm (float): Random subspace method. The percentage of features to use
            at each split selection.
    """
    # Type alias
    Objective: ClassVar[type] = Literal['Logloss', 'LDAM', 'Focal', 'LA']
    EvalMetric: ClassVar[type] = Literal['Logloss', 'Accuracy', 'F1', 'Recall', 'AUC', 'PRAUC:use_weights=false', 'BrierScore', 'MCC', 'WKappa']
    Depth: ClassVar[type] = Annotated[int, Field(ge=1, le=16)]
    L2LeafReg: ClassVar[type] = NonNegativeFloat
    LearningRate: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]
    RSM: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]
    ScalePosWeight: ClassVar[type] = PositiveFloat

    # Params
    objective: Objective = 'LDAM'
    eval_metric: EvalMetric = 'PRAUC:use_weights=false'
    depth: Depth | tuple[Depth, Depth] | list[Depth] = (6, 10)
    l2_leaf_reg: L2LeafReg | tuple[L2LeafReg, L2LeafReg] | list[L2LeafReg] = (0., 1000.)
    learning_rate: LearningRate | tuple[LearningRate, LearningRate] | list[LearningRate] = (0.001, 0.1)
    rsm: RSM | tuple[RSM, RSM] | list[RSM] = (0.1, 1.)
    scale_pos_weight: ScalePosWeight | tuple[ScalePosWeight, ScalePosWeight] | list[ScalePosWeight] = 6.

    # Custom Loss Params
    LDAM_c: NonNegativeFloat | tuple[NonNegativeFloat, NonNegativeFloat] = 1.0
    Focal_gamma: NonNegativeFloat | tuple[NonNegativeFloat, NonNegativeFloat] = (1.0, 3.0)
    LA_tau: NonNegativeFloat = 1.0

    model_config = ConfigDict(extra='forbid')


class CBModelConfig(BaseModel):
    """CatBoost Model configuration.

    Args:
        early_stopping_rounds (int): If provided, stop training if one
            metric of one validation data doesn't improve in the last
            `early_stopping_round` rounds. Defaults to 100.
        fold_count (int): Number of different cross validation folds to train.
            Defaults to 5.
        num_boost_round (int): Number of boosting iterations. Defaults to 10000.
        seed (int): The random seed used for training. Defaults to 42.
        verbose (Union[bool, int]): show log for training. Defaults to 100.
    """

    params: CBParams = CBParams()
    early_stopping_rounds: int = 100
    nfold: int = 5
    num_boost_round: int = 10_000
    seed: int = 42
    stratified: bool = True
    verbose: Union[bool, int] = False

    model_config = ConfigDict(extra='forbid')


class XGBParams(BaseModel):
    """XGBoost model training parameters. For details, see
    [XGBoost's documentation](
    https://xgboost.readthedocs.io/en/stable/parameter.html).

    Args:
        objective (Literal): Loss function in training. Defaults to Logloss.
        eval_metric (Literal): Metric to compare models. Defaults to Accuracy.
        max_depth (int): Limit the max depth for tree model.
        reg_lambda (float): Coefficient of the L2 regularization term.
        learning_rate (float): Learning rate.
        subsample (float): Random subspace method. The percentage of features
            to use at each split selection.
    """
    # Type alias
    Objective: ClassVar[type] = Literal['binary:logistic', 'LDAM', 'Focal', 'LA']
    EvalMetric: ClassVar[type] = Literal['logloss', 'error', 'auc', 'aucpr']
    MaxDepth: ClassVar[type] = Annotated[int, Field(ge=1, le=16)]
    RegLambda: ClassVar[type] = NonNegativeFloat
    LearningRate: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]
    ScalePosWeight: ClassVar[type] = NonNegativeFloat
    SubSample: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]

    # Params/Defining defaults
    objective: Objective = 'LA'
    eval_metric: EvalMetric = 'aucpr'
    max_depth: MaxDepth | tuple[MaxDepth, MaxDepth] | list[MaxDepth] = (6, 10)
    reg_lambda: RegLambda | tuple[RegLambda, RegLambda] | list[RegLambda] = (0., 1000.)
    learning_rate: LearningRate | tuple[LearningRate, LearningRate] | list[LearningRate] = (0.001, 0.1)
    scale_pos_weight: ScalePosWeight | tuple[ScalePosWeight, ScalePosWeight] | list[ScalePosWeight] = 6.
    subsample: SubSample | tuple[SubSample, SubSample] | list[SubSample] = (0.1, 1.)
    
    # Custom Loss Params
    LDAM_c: NonNegativeFloat | tuple[NonNegativeFloat, NonNegativeFloat] = (1.0, 10.0)
    Focal_gamma: NonNegativeFloat | tuple[NonNegativeFloat, NonNegativeFloat] = (1.0, 3.0)
    LA_tau: NonNegativeFloat = 1.0

    model_config = ConfigDict(extra='forbid')


class XGBModelConfig(BaseModel):
    """XGBoost Model configuration.

    Args:
        early_stopping_rounds (int): If provided, stop training if one
            metric of one validation data doesn't improve in the last
            `early_stopping_round` rounds. Defaults to 100.
        nfold (int): Number of different cross validation folds to train.
            Defaults to 5.
        num_boost_round (int): Number of boosting iterations. Defaults to 10000.
        seed (int): The random seed used for training. Defaults to 42.
        verbosity (Union[bool, int]): show log for training. Defaults to 100.
    """

    params: XGBParams = XGBParams()
    early_stopping_rounds: int = 100
    nfold: int = 5
    num_boost_round: int = 10_000
    seed: int = 42
    stratified: bool = True
    verbose_eval: Union[bool, int] = False

    model_config = ConfigDict(extra='allow')


class LGBParams(BaseModel):
    """
    LightGBM model training parameters. For details, see
    [LightGBM's documentation](
    https://lightgbm.readthedocs.io/en/stable/Parameters.html).
    """
    # Type alias
    Objective: ClassVar[type] = Literal['binary', 'LDAM', 'Focal', 'LA']
    Metric: ClassVar[type] = Literal['average_precision']
    MaxDepth: ClassVar[type] = Annotated[int, Field(ge=1, le=16)]
    RegLambda: ClassVar[type] = NonNegativeFloat
    LearningRate: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]
    ScalePosWeight: ClassVar[type] = NonNegativeFloat
    SubSample: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]

    # Params/Defining defaults
    objective: Objective = 'LA'
    metric: Metric = 'average_precision'
    max_depth: MaxDepth | tuple[MaxDepth, MaxDepth] | list[MaxDepth] = (6, 10)
    num_leaves: NonNegativeInt = 31
    reg_lambda: RegLambda | tuple[RegLambda, RegLambda] | list[RegLambda] = (0., 1000.)
    learning_rate: LearningRate | tuple[LearningRate, LearningRate] | list[LearningRate] = (0.001, 0.1)
    scale_pos_weight: ScalePosWeight | tuple[ScalePosWeight, ScalePosWeight] | list[ScalePosWeight] = 6.
    subsample: SubSample | tuple[SubSample, SubSample] | list[SubSample] = (0.1, 1.)
    verbosity: int = -1

    # Custom Loss Params
    LDAM_c: NonNegativeFloat | tuple[NonNegativeFloat, NonNegativeFloat] = (1.0, 10.0)
    Focal_gamma: NonNegativeFloat | tuple[NonNegativeFloat, NonNegativeFloat] = (1.0, 3.0)
    LA_tau: NonNegativeFloat = 1.0

    model_config = ConfigDict(extra='forbid')


class LGBModelConfig(BaseModel):
    """LightGBM Model configuration."""

    params: LGBParams = LGBParams()
    callbacks: list = [lgb.early_stopping(stopping_rounds=100)]
    nfold: int = 5
    num_boost_round: int = 10_000
    seed: int = 42
    stratified: bool = True

    model_config = ConfigDict(extra='allow')
