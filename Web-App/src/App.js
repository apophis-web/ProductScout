import React, { useState } from 'react';
import Graph from './graph';
import './App.css';
import './login_page.css';

function App() {
  const [showLogin, setShowLogin] = useState(false);

  const handleTryProductScout = () => {
    setShowLogin(true);
  };

  const handleLoginClose = () => {
    setShowLogin(false);
  };

  return (
    <div className="App">
      <div className="header">
        <h1 className="title">ProductScout</h1>
        <button className="about-us-button">About Us</button>
        <button className="payment-plan-button">Payment Plan</button>
        <button className="login-button" onClick={handleTryProductScout}>
          Login/Signup
        </button>
      </div>

      {!showLogin ? (
        <div>
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
            <img src={require('./images/mongodb.svg')} className="mongo" alt="mongo" />
            <img src={require('./images/aws.png')} className="aws" alt="aws" />
            <img src={require('./images/react.png')} className="react" alt="react" />
          </div>
        </div>
      ) : (
        <div className="login-page">
          <button className="close-button" onClick={handleLoginClose}>
            X
          </button>
          <h2>Login to ProductScout</h2>
          <form>
            <label>
              Email:
              <input type="email" name="email" required />
            </label>
            <br />
            <label>
              Password:
              <input type="password" name="password" required />
            </label>
            <br />
            <button type="submit">Login</button>
          </form>
        </div>
      )}
    </div>
  );
}

export default App;
