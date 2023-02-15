import React from 'react';
import './login_page.css';

function LoginPage() {
  return (
    <div className="login-page">
      <h2>Login to ProductScout</h2>
      <form>
        <label htmlFor="email">Email:</label>
        <input type="email" id="email" name="email" required />

        <label htmlFor="password">Password:</label>
        <input type="password" id="password" name="password" required />

        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default LoginPage;
