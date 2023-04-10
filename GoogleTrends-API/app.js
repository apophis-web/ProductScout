const express = require('express');

const cors = require('cors');


const authRoutes = require('./routes/auth');

const app = express();

app.get('/', (req,res ) =>{
    res.send("We are the Best")
});


//Middlewares
app.use(cors());
app.use(express.json());
app.use('/', authRoutes);


app.listen(3000);
