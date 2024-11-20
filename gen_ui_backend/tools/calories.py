from typing import Dict, Union

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool


class CaloriesInput(BaseModel):
    weight: float = Field(..., description="Weight in kg")
    height: float = Field(..., description="Height in cm")
    age: int = Field(..., description="Age in years")
    gender: str = Field(..., description="Gender")
    activity: str = Field(..., description="Activity level")
    target: str = Field(..., description="Target body weight")


@tool("calories-calculator", args_schema=CaloriesInput, return_direct=True)
def calories_calculator(
    weight: float, height: float, age: int, gender: str, activity: str, target: str
) -> Dict[str, Union[str, float]]:
    """Calculate the amount of calories required to maintain a healthy weight."""

    bmi = round(weight / ((height / 100) ** 2), 2)

    if bmi < 18.5:
        cathegory = "Underweight"
    elif 18.5 <= bmi < 25:
        cathegory = "Normal weight"
    elif 25 <= bmi < 30:
        cathegory = "Overweight"
    else:
        cathegory = "Obese"

    if gender.lower() not in [
        "male",
        "female",
        "laki laki",
        "laki-laki",
        "pria",
        "wanita",
        "perempuan",
    ]:
        raise ValueError("Invalid gender. Must be 'male' or 'female'.")

    if activity.lower() not in [
        "sedentary",
        "tidak aktif",
        "tidak banyak bergerak",
        "lightly active",
        "sedikit aktif",
        "moderately active",
        "cukup aktif",
        "aktif",
        "very_active",
        "sangat aktif",
        "extremely active",
        "sangat aktif sekali",
    ]:
        raise ValueError(
            "Invalid activity level. Must be 'sedentary', 'lightly active', 'moderately active', 'very active', or 'extremely active'."
        )

    if target.lower() not in [
        "maintain",
        "mempertahankan",
        "gain",
        "meningkatkan",
        "menaikkan",
        "loss",
        "menurunkan",
    ]:
        raise ValueError("Invalid target. Must be 'maintain', 'gain', or 'loss'.")

    if (
        gender.lower() == "male"
        or gender.lower() == "laki laki"
        or gender.lower() == "pria"
        or gender.lower() == "laki-laki"
    ):
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif (
        gender.lower() == "female"
        or gender.lower() == "wanita"
        or gender.lower() == "perempuan"
    ):
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    if (
        activity.lower() == "sedentary"
        or activity.lower() == "tidak banyak bergerak"
        or activity.lower() == "tidak aktif"
    ):
        tdee = bmr * 1.2
    elif activity.lower() == "lightly active" or activity.lower() == "sedikit aktif":
        tdee = bmr * 1.375
    elif (
        activity.lower() == "moderately active"
        or activity.lower() == "cukup aktif"
        or activity.lower() == "aktif"
    ):
        tdee = bmr * 1.55
    elif activity.lower() == "very active" or activity.lower() == "sangat aktif":
        tdee = bmr * 1.725
    elif (
        activity.lower() == "extremely active"
        or activity.lower() == "sangat aktif sekali"
    ):
        tdee = bmr * 1.9

    if target.lower() == "maintain" or target.lower() == "mempertahankan":
        required_calories = tdee
    elif (
        target.lower() == "gain"
        or target.lower() == "meningkatkan"
        or target.lower() == "menaikkan"
    ):
        required_calories = tdee + 500
    elif target.lower() == "loss" or target.lower() == "menurunkan":
        required_calories = tdee - 500

    return {
        "bmi": round(bmi, 2),
        "cathegory": cathegory,
        "bmr": round(bmr, 2),
        "maintenance_calories": round(tdee, 2),
        "required_calories": round(required_calories, 2),
    }
