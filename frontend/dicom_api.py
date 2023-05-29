import os
from pathlib import Path
import dotenv
from googleapiclient import discovery
import pydicom
from pydicom.tag import Tag

import atexit
from dataclasses import dataclass

dotenv.load_dotenv()

API_VERSION = "v1"
SERVICE_NAME = "healthcare"
# Returns an authorized API client by discovering the Healthcare API
# and using GOOGLE_APPLICATION_CREDENTIALS environment variable.
SERVICE = discovery.build(SERVICE_NAME, API_VERSION)


def exit_handler(*args, **kwargs):
    print("Handling exit...")
    SERVICE.close()


# Ensure service is almost guaranteed to be closed upon exit (from https://stackoverflow.com/a/72592788)
atexit.register(exit_handler)
# signal.signal(signal.SIGTERM, exit_handler)
# signal.signal(signal.SIGINT, exit_handler)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
LOCATION = os.environ["DICOM_LOCATION"]
DATASET_ID = os.environ["DICOM_DATASET_ID"]
DICOM_STORE_ID = os.environ["DICOM_STORE_ID"]

OUTPUT_DIR = Path("downloads")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class DICOMObjectType:
    INSTANCES = "instances"
    SERIES = "series"
    STUDIES = "studies"


class _GoodResponse:
    ok: bool = True
    reason: str = "Ok (MOCK)"
    status_code: int = 200
    

def _dicom_dataset_parent(project_id: str, location: str):
    return f"projects/{project_id}/locations/{location}"


def _dicom_store_name(project_id: str, location: str, dataset_id: str, dicom_store_id: str):
    return f"projects/{project_id}/locations/{location}/datasets/{dataset_id}/dicomStores/{dicom_store_id}"


def human_readable_instance(instance_dict: dict) -> dict:
    ds = pydicom.Dataset()
    
    for k, v in instance_dict.items():
        # First convert all keys to pydicom Tags: https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.tag.Tag.html#pydicom.tag.Tag
        tag = Tag(k)
        # Get Value Representation (2-character string that barely describes what this bit of info actually is)
        try:
            VR = v["vr"]
        except KeyError:
            print(f"WARN: tag {tag} had no VR (dict = {v})")
            continue
        # Get the actual value of this datum
        try:
            value = v["Value"]
        except KeyError:
            print(f"WARN: tag {tag} with VR {VR} had no value attribute (dict = {v})")
            continue
        ds.add_new(tag=tag, VR=VR, value=value)

    return {
        "SOPClassUID": ds.get("SOPClassUID"),
        "SOPInstanceUID": ds.get("SOPInstanceUID"),
        "SeriesInstanceUID": ds.get("SeriesInstanceUID"),
        "StudyDate": ds.get("StudyDate"),
        "StudyID": ds.get("StudyID"),
        "StudyInstanceUID": ds.get("StudyInstanceUID"),
        "StudyTime": ds.get("StudyTime")
    }


def list_datasets(project_id: str = PROJECT_ID, location: str = LOCATION):
    """Lists the datasets in the project.

    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets
    before running the sample."""
    dataset_parent = _dicom_dataset_parent(project_id, location)

    datasets = (
        SERVICE.projects()
        .locations()
        .datasets()
        .list(parent=dataset_parent)
        .execute()
        .get("datasets", [])
    )

    return datasets


def list_store_contents(project_id: str = PROJECT_ID, location: str = LOCATION, dataset_id: str = DATASET_ID, dicom_store_id: str = DICOM_STORE_ID, contents_type: str = DICOMObjectType.INSTANCES):
    """Handles the GET requests specified in DICOMweb standard.

    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom
    before running the sample."""
    parent = _dicom_store_name(project_id, location, dataset_id, dicom_store_id)
    dicom_web_path = contents_type
    
    request = SERVICE.projects().locations().datasets().dicomStores().searchForInstances(parent=parent, dicomWebPath=dicom_web_path)
    request.headers = {"Content-Type": "application/dicom+json; charset=utf-8"}
    response = request.execute()

    return response


def dicomweb_retrieve_instance(
    study_uid: str,
    series_uid: str,
    instance_uid: str,
    project_id: str = PROJECT_ID,
    location: str = LOCATION,
    dataset_id: str = DATASET_ID,
    dicom_store_id: str = DICOM_STORE_ID,
):
    """Handles the GET requests specified in the DICOMweb standard.

    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom
    before running the sample."""
    output_file = OUTPUT_DIR / f"{instance_uid}.dcm"
    if output_file.exists():
        print(f"{output_file} already exists, so not downloading file again")
        return output_file, _GoodResponse

    # Imports Python's built-in "os" module
    import os

    # Imports the google.auth.transport.requests transport
    from google.auth.transport import requests

    # Imports a module to allow authentication using a service account
    from google.oauth2 import service_account

    # Gets credentials from the environment.
    credentials = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )
    # Creates a requests Session object with the credentials.
    session = requests.AuthorizedSession(scoped_credentials)

    # URL to the Cloud Healthcare API endpoint and version
    base_url = "https://healthcare.googleapis.com/v1"

    url = f"{base_url}/projects/{project_id}/locations/{location}"

    dicom_store_path = "{}/datasets/{}/dicomStores/{}".format(
        url, dataset_id, dicom_store_id
    )

    dicomweb_path = "{}/dicomWeb/studies/{}/series/{}/instances/{}".format(
        dicom_store_path, study_uid, series_uid, instance_uid
    )

    # Set the required Accept header on the request
    headers = {"Accept": "application/dicom; transfer-syntax=*"}
    response = session.get(dicomweb_path, headers=headers)
    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)
        print(f"Retrieved DICOM instance and saved to {output_file} in current directory")

    return output_file, response