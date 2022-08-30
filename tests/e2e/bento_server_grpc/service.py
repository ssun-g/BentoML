from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from pydantic import BaseModel
from _servicer import TestServiceServicer
from PIL.Image import fromarray

import bentoml
from bentoml.io import File
from bentoml.io import JSON
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray
from bentoml.io import PandasDataFrame
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from PIL.Image import Image as PILImage
    from numpy.typing import NDArray

    from bentoml.grpc.v1alpha1 import service_test_pb2 as pb_test
    from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services_test
    from bentoml._internal.types import FileLike
    from bentoml._internal.types import JSONSerializable
    from bentoml.picklable_model import get_runnable
    from bentoml._internal.runner.runner import RunnerMethod

    RunnableImpl = get_runnable(bentoml.picklable_model.get("py_model.case-1.grpc.e2e"))

    class PythonModelRunner(bentoml.Runner):
        predict_file: RunnerMethod[RunnableImpl, [list[FileLike[bytes]]], list[bytes]]
        echo_json: RunnerMethod[
            RunnableImpl, [list[JSONSerializable]], list[JSONSerializable]
        ]
        echo_ndarray: RunnerMethod[RunnableImpl, [NDArray[t.Any]], NDArray[t.Any]]
        double_ndarray: RunnerMethod[RunnableImpl, [NDArray[t.Any]], NDArray[t.Any]]
        multiply_float_ndarray: RunnerMethod[
            RunnableImpl,
            [NDArray[np.float32], NDArray[np.float32]],
            NDArray[np.float32],
        ]
        double_dataframe_column: RunnerMethod[
            RunnableImpl, [pd.DataFrame], pd.DataFrame
        ]

else:
    from bentoml.grpc.utils import import_generated_stubs

    pb_test, services_test = import_generated_stubs(file="service_test.proto")
    np = LazyLoader("np", globals(), "numpy")
    pd = LazyLoader("pd", globals(), "pandas")


py_model = t.cast(
    "PythonModelRunner",
    bentoml.picklable_model.get("py_model.case-1.grpc.e2e").to_runner(),
)

svc = bentoml.Service(name="general_grpc_service.case-1.e2e", runners=[py_model])

services_name = [v.full_name for v in pb_test.DESCRIPTOR.services_by_name.values()]
svc.mount_grpc_servicer(
    TestServiceServicer,
    add_servicer_fn=services_test.add_TestServiceServicer_to_server,
    service_names=services_name,
)

# TODO: write custom interceptor
# from _interceptor import AsyncContextInterceptor
# svc.add_grpc_interceptor(AsyncContextInterceptor, usage="NLP", accuracy_score=0.8247)


@svc.api(input=JSON(), output=JSON())
async def echo_json(json_obj: JSONSerializable) -> JSONSerializable:
    batched = await py_model.echo_json.async_run([json_obj])
    return batched[0]


class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float


class IrisClassificationRequest(BaseModel):
    request_id: str
    iris_features: IrisFeatures


@svc.api(
    input=JSON(pydantic_model=IrisClassificationRequest),
    output=JSON(),
)
def echo_json_validate(input_data: IrisClassificationRequest) -> dict[str, float]:
    print("request_id: ", input_data.request_id)
    return input_data.iris_features.dict()


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def double_ndarray(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    return await py_model.double_ndarray.async_run(arr)


@svc.api(input=NumpyNdarray.from_sample(np.random.rand(2, 2)), output=NumpyNdarray())
async def echo_ndarray_from_sample(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    assert arr.shape == (2, 2)
    return await py_model.echo_ndarray.async_run(arr)


@svc.api(input=NumpyNdarray(shape=(2, 2), enforce_shape=True), output=NumpyNdarray())
async def echo_ndarray_enforce_shape(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    assert arr.shape == (2, 2)
    return await py_model.echo_ndarray.async_run(arr)


@svc.api(
    input=NumpyNdarray(dtype=np.float32, enforce_dtype=True), output=NumpyNdarray()
)
async def echo_ndarray_enforce_dtype(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    assert arr.dtype == np.float32
    return await py_model.echo_ndarray.async_run(arr)


@svc.api(input=File(), output=File())
async def predict_file(f: FileLike[bytes]) -> bytes:
    batch_ret = await py_model.predict_file.async_run([f])
    return batch_ret[0]


@svc.api(input=Image(), output=Image(mime_type="image/bmp"))
async def echo_image(f: PILImage) -> NDArray[t.Any]:
    assert isinstance(f, PILImage)
    return np.array(f)


@svc.api(
    input=PandasDataFrame(dtype={"col1": "int64"}, orient="columns"),
    output=PandasDataFrame(),
)
async def double_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    assert df["col1"].dtype == "int64"
    output = await py_model.double_dataframe_column.async_run(df)
    dfo = pd.DataFrame()
    dfo["col1"] = output
    return dfo


@svc.api(
    input=Multipart(original=Image(), compared=Image()),
    output=Multipart(r1=Image(), r2=Image()),
)
async def predict_multi_images(original: Image, compared: Image):
    output_array = await py_model.multiply_float_ndarray.async_run(
        np.array(original), np.array(compared)
    )
    img = fromarray(output_array)
    return {"r1": img, "r2": img}
