from pathlib import Path
from flask import Flask, render_template
import pydicom
import frontend.dicom_api as dicom_api
from frontend.dicom_video import convert_dicom_color, vidwrite

app = Flask(__name__)

@app.route("/")
def list_instances():
    instances = dicom_api.list_store_contents(contents_type=dicom_api.DICOMObjectType.INSTANCES)
    instances = [dicom_api.human_readable_instance(instance) for instance in instances]
    for instance in instances:
        study_UID = instance["StudyInstanceUID"].replace(".", "-")
        series_UID = instance["SeriesInstanceUID"].replace(".", "-")
        instance_UID = instance["SOPInstanceUID"].replace(".", "-")
        instance["formUrl"] = f"/instances/{study_UID}/{series_UID}/{instance_UID}"

    print(instances)
    return render_template("list_instances.html", instances=instances)

@app.route("/instances/retrieve/<string:study_uid>/<string:series_uid>/<string:instance_uid>")
def retrieve_instance(study_uid: str, series_uid: str, instance_uid: str):
    study_uid = study_uid.replace("-", ".")
    series_uid = series_uid.replace("-", ".")
    instance_uid = instance_uid.replace("-", ".")
    dicom_file, video_file, response = dicom_api.dicomweb_retrieve_instance(study_uid=study_uid, series_uid=series_uid, instance_uid=instance_uid)
    return {
        "outputFile": str(dicom_file),
        "responseOk": response.ok,
        "responseReason": response.reason,
        "responseStatus": response.status_code,
    }

@app.route("/instances/<string:study_uid>/<string:series_uid>/<string:instance_uid>")
def instance_form_view(study_uid: str, series_uid: str, instance_uid: str):
    study_uid = study_uid.replace("-", ".")
    series_uid = series_uid.replace("-", ".")
    instance_uid = instance_uid.replace("-", ".")
    dicom_file, video_file, response = dicom_api.dicomweb_retrieve_instance(study_uid=study_uid, series_uid=series_uid, instance_uid=instance_uid)
    return render_template("instance_form_view.html", video_filepath=str(video_file))
