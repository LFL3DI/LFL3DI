import * as Vue from  './vue.esm-browser.js'
const App = {
    created() {
        if (ws.readyState === 1) {
            this.loaded = true
        } else {
            ws.onopen = () => {
                this.loaded = true
            }
        }

        setInterval(()=>{
            let message = JSON.stringify({ cmd: 'read' });
            console.log("Sending WebSocket message:", message);
            ws.send(message);
        }, 1000)
        ws.onmessage = function (event) {
            var data = JSON.parse(event.data);
            console.log("Received WebSocket data:", data);
        
            if (data.depth) {
                document.getElementById("amplitudeView").src = "data:image/png;base64," + data.depth;
            }
        }
    },
    mounted(){
        document.getElementById("mode1").addEventListener("click", () => {
            if (ws.readyState === WebSocket.OPEN) {
                console.log("Sending WebSocket message: live mode");
                ws.send(JSON.stringify({ cmd: "live" }));
            }
        });
        document.getElementById("mode2").addEventListener("click", () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: "Continuous" }));
            }
        });
        document.getElementById("mode3").addEventListener("click", () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: "triggering" }));
            }
        });
        document.getElementById("stop").addEventListener("click", () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: "stop" }));
            }
        });
    },
    data() {
        return {
            loaded: false,
        }
    },
    computed: {
    }
}

const app = Vue.createApp(App);
app.mount("#app");