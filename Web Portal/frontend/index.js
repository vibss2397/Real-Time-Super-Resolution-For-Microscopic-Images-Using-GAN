var express = require('express');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io')(server);
var request=require('request');
var fs = require('fs');
var path = require('path');
var bodyParser = require('body-parser');
server.listen(3001);

// WARNING: app.listen(80) will NOT work here!
var client_list = [];

var port_mobile = 'http://192.168.43.250'
var port_wifi = 'http://192.168.1.9'
var port_final;

app.set('views', path.join(__dirname, "views"))
app.set('view engine', 'hbs');
app.use(express.static('views/images')); 

app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())

var download = function(uri, filename, callback){
  request.head(uri, function(err, res, body){    
    request(uri).pipe(fs.createWriteStream(filename)).on('close', callback);
  });
};


app.get('/', function (req, res) {
  request(port_wifi+':5000/').pipe(fs.createWriteStream('doodle2.png'));
  res.render('index', {im: 'doodle2.png'})
}); 

app.get('/app', function (req, res) {
  res.render('app', {im: 'doodle.png'})
}); 

app.post('/crop', function(req, res){
  console.log(req.body)
  coord={x1: req.body.x1, x2: req.body.x2, y1: req.body.y1, y2: req.body.y2, w: req.body.w, h: req.body.h}
  dic = {domain: port_wifi+':3001/'+req.body.image, coordinate: coord};
  console.log('asd');
  request.post({url: port_wifi+':5000/send_request', form: JSON.stringify(dic)}, 
  function(err,httpResponse,body){
    res.send(body);
  })
});
app.post('/set_ip', function(req, res){
  port_final = req.body.ip
})
app.post('/rename', function(req, res){
  if(fs.existsSync('./views/images/temp2.png')){
  fs.unlink('./views/images/temp.png', (err) => {
    if (err) {throw err;res.send('-1');}
    fs.renameSync('./views/images/temp2.png', './views/images/temp.png');
  });
  fs.unlink('./views/images/temp3.png', (err) => {
    if (err) {throw err;res.send('-1');}
    fs.renameSync('./views/images/temp4.png', './views/images/temp3.png');
    res.send('1');
  });
}else{
  res.send('1')
}
});

app.post('/convert', function(req, res){
  request.post({url: port_wifi+':5000/upsample', form: JSON.stringify(req.body)}, function(err, httpResponse, body){
    res.send(body);
  })
  
});

io.on('connection', function (socket) {
    client_list.push(socket.id)
    console.log(client_list);
    io.emit('test', {asd: 'asddd'})
    socket.on('firse connected', (msg)=>{
        console.log(msg);
    })
    socket.on('firse disconnected', (msg)=>{
      client_list.pop();
      console.log(msg);
  })
  socket.on('sending_data', (msg)=>{
    console.log('recieved');
    try {
      if (fs.existsSync('./views/images/temp.png') || (fs.existsSync('./views/images/temp.png')&&fs.existsSync('./views/images/temp2.png'))) {
        console.log('doing this 2');
        var bas64 = msg.replace('/^data:image\/png;base64,/', "");
        fs.writeFile('./views/images/temp2.png', bas64, 'base64', function(err) {
          console.log(err);
        });
        dic = {link: port_wifi+':3001/temp2.png'}
        request.post({url: port_wifi+':5000/resize', form: JSON.stringify(dic)}, function(err,httpResponse,body){
          console.log('yas');
          request(body).pipe(fs.createWriteStream('./views/images/temp4.png')); 
          io.emit('send', {image: 'temp3.png'})
        })
        
      } else{
        var bas64 = msg.replace('/^data:image\/png;base64,/', "");
        fs.writeFile('./views/images/temp.png', bas64, 'base64', function(err) {
          console.log(err);
        });
        dic = {link: port_wifi+':3001/temp.png'}
        request.post({url: port_wifi+':5000/resize', form: JSON.stringify(dic)}, function(err,httpResponse,body){
          console.log('yas');
          request(body).pipe(fs.createWriteStream('./views/images/temp3.png')); 
          io.emit('send', {image: 'temp3.png'})
        })
        
        console.log('yaaha');
       }
    } catch(err) {
      console.error(err)
    }
    // io.emit('send', {image: msg})
  })
});
