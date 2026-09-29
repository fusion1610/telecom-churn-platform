from pydantic import BaseModel, ConfigDict, Field


class ChurnPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CRM_PID_Value_Segment: str
    EffectiveSegment: str

    Active_subscribers: float = Field(ge=0)
    Not_Active_subscribers: float = Field(ge=0)
    Suspended_subscribers: float = Field(ge=0)
    Total_SUBs: float = Field(gt=0)

    AvgMobileRevenue: float = Field(ge=0)
    AvgFIXRevenue: float = Field(ge=0)
    ARPU: float = Field(ge=0)

    TotalRevenue: float | None = Field(
        default=None,
        ge=0,
    )


class ChurnPredictionResponse(BaseModel):
    Churn_Probability: float = Field(ge=0, le=1)
    Retention_Flag: bool
    Risk_Band: str

    Revenue_at_Risk: float | None = None
    Priority_Tier: str | None = None
    Retention_Action: str | None = None
    Retention_Urgency: str | None = None