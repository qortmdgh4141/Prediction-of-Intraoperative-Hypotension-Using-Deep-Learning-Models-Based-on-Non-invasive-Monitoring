// written by Hyung-Chul Lee (vital@snu.ac.kr)
// Download minimum_vitalserver.js from the github repository.
// Please download and nodejs server (LTS version) from https://nodejs.org.
// In the folder containing minimum_vitalserver.js, run "npm install socket.io@2 express".
// Execute "node minimum_vitalserver.js".
// Allow access when the warning pops up.
// Please add "SERVER_IP=127.0.0.1:3000" to vr.conf.
// Run Vital Recorder and Add Device (eg. Demo)
// Open the browser and connect to "http://localhost:3000".
const app = require('express')();
const server = require('http').createServer(app);
const io = require("socket.io")(server);
const zlib = require('zlib');

app.get('/', (req, res) => {
    res.send(`<script src="/socket.io/socket.io.js"></script>
            <div id=display></div>
            <script>
            io().on('send_data', (data) => {
                var rooms = data['rooms'];
                var trks = rooms[0]['trks'];
                var str = '';
                for (var i in trks) {
                    str += trks[i].name + '=';
                    if (trks[i].recs.length > 0)
                        str += trks[i].recs[0].val;
                    str += '<br>';
                }
                document.getElementById('display').innerHTML = str;
                //console.log(trks);
            });
        </script>`);
});

io.on('connection', (socket) => {
    console.log(`connect ${socket.id}`);
    socket.on("disconnect", (reason) => {
      console.log(`disconned ${socket.id} due to ${reason}`);
    });
    socket.on("send_data", data => {
        console.log("data received");
        try{
            var z1 = zlib.inflateSync(Buffer.from(data, 'binary')).toString();
            z1.replace('/[\x00-\x1F\x7F]/u', '');
            z1.replace('nan', '""');
            var json = JSON.parse(z1);
            socket.broadcast.emit('send_data', json);
        } catch(err){
            console.log("send_data ERROR", err, data);
        }
    });
});

server.listen(3000, () => {
  console.log('listening on *:3000');
});