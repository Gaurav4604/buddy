import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Question from "./pages/Question";
import Upload from "./pages/Upload";
import Insights from "./pages/Insights";
import Profile from "./pages/Profile";
import Header from "./components/Header";

const App: React.FC = () => {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/question" element={<Question />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/insights" element={<Insights />} />
        <Route path="/profile" element={<Profile />} />
      </Routes>
    </Router>
  );
};

export default App;
