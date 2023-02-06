import d3 from "./images/d3.png"
import pyth from "./images/python.png"
import pytorch from "./images/pytorch.png"
import node from "./images/node.png"
import skit from "./images/skit.png"
import selenium from "./images/selenium.png"
import mongo from "./images/mongodb.svg"
import aws from "./images/aws.png"
import react from "./images/react.png"
import Graph from './graph';

import './App.css';
// import './login_page.css';

function App() {
  return (
    <div className="App">
      <div className="header">
        <h1 className="title">ProductScout</h1>
        {/* <h1 className="about-us">About Us</h1> */}
      </div>

    
      <div className="graph"> 
        <Graph />
      </div>
      
      
      <div className="overview">
        Welcome to ProductScout - your one-stop solution for e-commerce trend forecasting. 
        With our advanced recommendation system, we use publicly available data to provide insightful predictions on future trends in the market. 
        Our goal is to empower e-commerce businesses to make informed decisions and stay ahead of the curve. 
        Join us on this journey as we revolutionize the e-commerce industry with data-driven solutions. 
        Start using ProductScout today and take your business to the next level!
      </div>

      <button className="try-productscout-button">Try ProductScout</button>
      <div className = "technologies">
        <h1>
          Technologies
        </h1>
      </div>

      <div className = "images">
        <img src = {d3} className = "D3"/>
        <img src = {pyth} className = "python"/>
        <img src = {pytorch} className = "pytorch"/>
        <img src = {node} className = "node"/>
        <img src = {skit} className = "skit"/>
        <img src = {selenium} className = "selenium"/>
        <img src = {mongo} className = "mongo"/>
        <img src = {aws} className = "aws"/>
        <img src = {react} className = "aws"/>
      </div>
    </div>

    
  );
}

export default App;