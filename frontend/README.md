# DICOM frontend
You'll need to install some extra dependencies for this:
- [pydicom](https://pydicom.github.io/pydicom/dev/index.html)
- [Flask](https://flask.palletsprojects.com/en/2.3.x/)
- [google-api-python-client](https://github.com/googleapis/google-api-python-client)
- [ffmpeg-python](https://pypi.org/project/ffmpeg-python/)

To install all of these, you can probably get away with `pip`:

```shell
pip install google-api-python-client Flask pydicom ffmpeg-python
```

You need a JSON file in the root of this repo to store the service account credentials. Go to "APIs and services > Credentials", click on our frontend service account near the bottom, click on the "Keys" tab, "Add key", "Create new key", and select the JSON option. Move the downloaded JSON file into the root of this repo.

You'll also need to add some environment variables to your `.env` file. Look near the top of `dicom_api.py` to see which environment variables are required.

To run the Flask server,
```shell
flask --app frontend.app run --debug
```

You can then navigate to http://localhost:5000/ to hopefully see a table of metadata for DICOM instances in our DICOM dataset!