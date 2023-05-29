async function retrieveDICOMFile(studyUID, seriesUID, instanceUID) {
    studyUIDEncoded = studyUID.replaceAll(".", "-");
    seriesUIDEncoded = seriesUID.replaceAll(".", "-");
    instanceUIDEncoded = instanceUID.replaceAll(".", "-");

    const downloadButtonCellID = `downloadButton-${instanceUIDEncoded}`;
    const downloadButtonCell = document.getElementById(downloadButtonCellID);
    const spinner = document.querySelector(`#${downloadButtonCellID} > .spinner-border`);
    const downloadButton = document.querySelector(`#${downloadButtonCellID} > button`);
    
    spinner.style.display = "block";
    downloadButton.disabled = true;

    const response = await fetch(`/instances/retrieve/${studyUIDEncoded}/${seriesUIDEncoded}/${instanceUIDEncoded}`);
    const jsonData = await response.json();
    downloadButtonCell.innerHTML = jsonData.outputFile;

    console.log(jsonData);
}