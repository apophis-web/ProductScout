import React from 'react';
import './login_page.css';

function LoginPage() {
  return (
    <div className="login-page">
      <div className='login-greet'>Login to ProductScout</div>
      <form>
        <label htmlFor="email">
          <div className='email-header'>Email</div>
        </label>
        <input type="email" id="email" name="email" required />

        <label htmlFor="password">
          <div className='password-header'>Password</div>

          </label>
        <input type="password" id="password" name="password" required />

        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default LoginPage;