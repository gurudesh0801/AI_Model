const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());

let isRecording = false;

// Start recording API
app.get('/start-recording', async (req, res) => {
  try {
    await axios.get('http://localhost:5001/start');
    isRecording = true;
    res.send('Recording started');
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Stop recording API
app.get('/stop-recording', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:5001/stop');
    isRecording = false;
    res.json(response.data);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

app.listen(5000, () => console.log('Server running on http://localhost:5000'));
