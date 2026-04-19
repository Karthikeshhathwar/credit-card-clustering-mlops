from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    BALANCE: float = Field(..., gt=0)
    PURCHASES: float = Field(..., ge=0)
    CASH_ADVANCE: float = Field(..., ge=0)

    CREDIT_LIMIT: float = Field(..., gt=0)

    PAYMENTS: float = Field(..., ge=0)
    MINIMUM_PAYMENTS: float = Field(..., ge=0)

    PURCHASES_TRX: int = Field(..., ge=0)
    CASHADVANCETRX: int = Field(..., ge=0)

    PRCFULLPAYMENT: float = Field(..., ge=0, le=1)
    TENURE: int = Field(..., ge=1)