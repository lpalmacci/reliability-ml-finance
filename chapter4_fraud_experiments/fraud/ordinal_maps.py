"""Order of the categories of the ordered variables of carclaims (fraud_oracle.csv).

Only the ordered variables are declared here; the other categorical variables are encoded by the
generic preprocessor. The categories are those found in the file; a category that does not appear in
the training part is encoded as -1.
"""
from __future__ import annotations

from fraud.specs import OrdinalSpec

CARCLAIMS_ORDINALS: tuple[OrdinalSpec, ...] = (
    OrdinalSpec(col="AgeOfVehicle",
                categories=("new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7")),
    OrdinalSpec(col="AgeOfPolicyHolder",
                categories=("16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50",
                            "51 to 65", "over 65")),
    OrdinalSpec(col="Days_Policy_Accident", categories=("none", "1 to 7", "8 to 15", "15 to 30", "more than 30")),
    OrdinalSpec(col="AddressChange_Claim",
                categories=("no change", "4 to 8 years", "2 to 3 years", "1 year", "under 6 months")),
    OrdinalSpec(col="Days_Policy_Claim", categories=("none", "8 to 15", "15 to 30", "more than 30")),
    OrdinalSpec(col="NumberOfCars", categories=("1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8")),
    OrdinalSpec(col="NumberOfSuppliments", categories=("none", "1 to 2", "3 to 5", "more than 5")),
    OrdinalSpec(col="PastNumberOfClaims", categories=("none", "1", "2 to 4", "more than 4")),
    OrdinalSpec(col="VehiclePrice",
                categories=("less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000",
                            "60000 to 69000", "more than 69000")),
)
