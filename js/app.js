import * as Vue from './vue.esm-browser.js'

const App = {
  created() {
    if (ws.readyState === 1) {
      this.loaded = true
    } else {
      ws.onopen = () => {
        this.loaded = true
      }
    }

    setInterval(() => {
      let message = JSON.stringify({ cmd: 'read' });
      // console.log("Sending WebSocket message:", message);
      ws.send(message);
    }, 1000);
  },

  mounted() {
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received WebSocket data:", data);

      if (data.depth) {
        document.getElementById("amplitudeView").src = "data:image/png;base64," + data.depth;
      }

      if (data.objects && Array.isArray(data.objects)) {
        const tableBody = document.getElementById("detectionTableBody");
        tableBody.innerHTML = "";

        data.objects.forEach(obj => {
          const row = document.createElement("tr");

          const indexCell = document.createElement("td");
          indexCell.textContent = obj.index;
          row.appendChild(indexCell);

          const timeCell = document.createElement("td");
          timeCell.textContent = obj.timestamp;
          row.appendChild(timeCell);

          const labelCell = document.createElement("td");
          labelCell.textContent = obj.label;
          row.appendChild(labelCell);

          const confCell = document.createElement("td");
          confCell.textContent = obj.confidence;
          row.appendChild(confCell);

          const xCell = document.createElement("td");
          const yCell = document.createElement("td");
          const zCell = document.createElement("td");
          xCell.textContent = obj.position[0];
          yCell.textContent = obj.position[1];
          zCell.textContent = obj.position[2];
          row.appendChild(xCell);
          row.appendChild(yCell);
          row.appendChild(zCell);

          const angleCell = document.createElement("td");
          angleCell.textContent = obj.angle + "°";
          row.appendChild(angleCell);

          tableBody.appendChild(row);
        });
      }
    };

    document.getElementById("recording").addEventListener("click", function () {
      console.log("Toggle state:", this.checked);

      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ cmd: this.checked ? "start" : "stop" }));
      }
    });

    document.getElementById("continuous").addEventListener("click", function () {
      console.log("Toggle state:", this.checked);

      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ cmd: "toggle_recording" }));
        ws.send(JSON.stringify({ cmd: this.checked ? "triggered" : "continuous" }));
      }
    });

    document.getElementById("objectType").addEventListener("change", function () {
      const selectedObject = this.value;
      console.log("Selected object type:", selectedObject);
  
      if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ cmd: "set_object", object: selectedObject }));
      }
    });
  },

  data() {
    return {
      loaded: false,
    }
  }
}

const app = Vue.createApp(App)
app.mount("#app")
