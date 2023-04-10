import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import Graph from './graph';
import './App.css';
import './login_page.css';
import './search_page.css';

const CustomizedLine = (props) => {
  const { stroke, points } = props;
  return (
    <g>
      <path
        d={`
          M${points[0].x},${points[0].y}
          ${points.map((p, i) => (i === 0 ? '' : `L${p.x},${p.y}`)).join('')}
        `}
        stroke={stroke}
        fill="none"
        strokeWidth="3"
      />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={5} fill={stroke} />
      ))}
    </g>
  );
};


function App() {
  const [data, setdata] = useState([]);
  const [data2, setdata2] = useState([]);

  const [showLogin, setShowLogin] = useState(false);
  const handleTryProductScout = () => {
    setShowLogin(true);
  };

  const handleLoginClose = () => {
    setShowLogin(false);
  };

  const [showSearchBar, setSearchBar] = useState(false);
  const handleSearchBar = () => {
    setSearchBar(true);
  };

  const handleCloseAndSearch = () => {
    handleSearchBar();
    handleLoginClose();
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      console.log('Enter key pressed');
      handleclick();
    }
  };

  const [searchQuery, setSearchQuery] = useState('');
  const [text, setText] = useState('');
  const [query, setQuery] = useState([]);

  const handleclick = () => {
    var data = {
      "name":text,
    }
    fetch('http://127.0.0.1:5000/search', {
      method: 'POST',
        headers: {
          'content-type': 'application/json'
      },
      body: JSON.stringify(data),
      }).then((response) => {
        response.json().then((body) => {
            console.log(body)
            fetch('http://127.0.0.1:5000/get_text', {
              method: 'POST',
                headers: {
                  'content-type': 'application/json'
              },
              body: JSON.stringify({"lists":body['main_list']}),
              }).then((response) => {
                response.json().then((body) => {
                  setdata(body["original"])
                  setdata2(body["pred"])
                  console.log(body)
                  var boxes = document.getElementsByClassName(
                    "chartContainer",
                  )
                  boxes[0].style.display = "block";
                })
              })
              
            setQuery(body["mapping"])
            var boxes = document.getElementsByClassName(
              'product-scout',
            )
            boxes[0].style.top = "20px";
            boxes[0].style.left = "20px";
            boxes[0].style.transform = "translate(0%, 0%)";
            boxes[0].style["font-size"] = "20px";

            var bar = document.getElementsByClassName(
              'search-input',
            )
            bar[0].style.top = "50px";
            bar[0].style.right = "130px";
            bar[0].style.transform = "translate(0%, 0%)";

            var search = document.getElementsByClassName(
              'search-button',
            )
            search[0].style.top = "45px";
            search[0].style.width = "100px";
            search[0].style.height = "60px";
            search[0].style.right = "25px";
            search[0].style.transform = "translate(0%, 0%)";
        });
      });
  };

  return (
    <div className="App">
      {showLogin && !showSearchBar &&(
        <div className="login-page">
          <div className='login-greet'>Login to ProductScout</div>
          <form>
            <label>
              <div className='email-header'>Email</div>
              <input type="email" name="email" required />
            </label>
            <br />
              <label>
                <div className='password-header'>Password</div>
                <input type="password" name="password" required />
              </label>
            <br />
            <button type="submit" onClick={handleCloseAndSearch} >Login</button>
          </form>
        </div>
      )}

      {!showLogin && !showSearchBar &&(
        <div>
          <div className="header">
            <h1 className="title">ProductScout</h1>
            <button className="about-us-button">About Us</button>
            <button className="payment-plan-button">Payment Plan</button>
            <button className="login-button" onClick={handleTryProductScout}>
              Login/Signup
            </button>
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

          <button className="try-productscout-button" onClick={handleTryProductScout}>
            Try ProductScout
          </button>

          <div className="technologies">
            <h1>Technologies</h1>
          </div>

          <div className="images">
            <img src={require('./images/d3.png')} className="D3" alt="d3" />
            <img src={require('./images/python.png')} className="python" alt="python" />
            <img src={require('./images/pytorch.png')} className="pytorch" alt="pytorch" />
            <img src={require('./images/node.png')} className="node" alt="node" />
            <img src={require('./images/skit.png')} className="skit" alt="skit" />
            <img src={require('./images/selenium.png')} className="selenium" alt="selenium" />
            <img src={require('./images/mongodb.png')} className="mongo" alt="mongo" />
            <img src={require('./images/aws.png')} className="aws" alt="aws" />
            <img src={require('./images/react.png')} className="react" alt="react" />
          </div>
        </div>
      )}

      {!showLogin && showSearchBar && (
        <div className="search-container">
          <input
            type="text"
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyPress={handleKeyPress}
            className="search-input"
          />

          <button onClick={handleclick} className="search-button">
            Search
          </button>

          <div className="product-scout">
            <h1>ProductScout</h1>
          </div>

          <div className='queryresults'>
            {query.length !== 0 ? <p className='queryitem'>Categoric Mapping: </p>: <div></div>}
            {query.map((item) => (
              <div>
                <p className='queryitem'> {item} </p>
              </div> 
            ))}
          </div> 

          <div className="chartContainer">
            <LineChart width={1000} height={500} data={data} backgroundColor="white">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="y"
                stroke="red"
                activeDot={{ r: 8 }}
                strokeWidth={3}
                isAnimationActive
                animationDuration={1500}
                content={<CustomizedLine />}
              />
            </LineChart>
          </div> 
        </div>
      )}
    </div>
  );
}

export default App;