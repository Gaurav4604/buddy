import React from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  MenuItem,
  Select,
  IconButton,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
// import { useNavigate } from "react-router-dom";

const Header: React.FC = () => {
  //   const navigate = useNavigate();
  const [topic, setTopic] = React.useState("topic1");

  const handleTopicChange = (event: unknown) => {
    setTopic(
      (event as React.ChangeEvent<{ value: unknown }>).target.value as string
    );
  };

  return (
    <AppBar position="static" sx={{ mb: 2 }}>
      <Toolbar>
        <Select
          value={topic}
          onChange={handleTopicChange}
          displayEmpty
          sx={{ color: "white", mr: 2 }}
        >
          <MenuItem value="topic1">Topic 1</MenuItem>
          <MenuItem value="topic2">Topic 2</MenuItem>
        </Select>

        <IconButton color="inherit" onClick={() => alert("Add new topic")}>
          <AddIcon />
        </IconButton>

        <Typography variant="h6" sx={{ flexGrow: 1, textAlign: "end" }}>
          Buddy
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
