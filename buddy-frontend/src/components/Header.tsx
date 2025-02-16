import React, { useEffect, useState } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  MenuItem,
  Select,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  SelectChangeEvent,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import { useDispatch, useSelector } from "react-redux";
import {
  setTopics,
  setTopicContent,
  addTopic,
  setActiveTopic,
} from "../store/topicAndContentSlice";
import { RootState } from "../store";
import { createTopic, getTopicContent, getTopics } from "../api/topic";

const Header: React.FC = () => {
  const dispatch = useDispatch();
  const topics = useSelector((state: RootState) => state.tnc.topics);
  const activeTopic = useSelector((state: RootState) => state.tnc.activeTopic);

  const [openDialog, setOpenDialog] = useState<boolean>(false);
  const [newTopicName, setNewTopicName] = useState<string>("");

  // Fetch topics on mount and set the first one as active if available.
  useEffect(() => {
    getTopics()
      .then((data) => {
        console.log("Fetched topics:", data);
        dispatch(setTopics(data));
        if (data.length > 0) {
          dispatch(setActiveTopic(data[0]));
        }
      })
      .catch((error) => {
        console.error("Error fetching topics", error);
      });
  }, [dispatch]);

  // When activeTopic changes, fetch its content.
  useEffect(() => {
    if (activeTopic && activeTopic.topic_name) {
      getTopicContent(activeTopic.topic_name)
        .then((data) => {
          dispatch(setTopicContent(data));
        })
        .catch((error) => {
          console.error("Error fetching topic content", error);
        });
    }
  }, [activeTopic, dispatch]);

  // When a new topic is selected, update local state and activeTopic in Redux.
  const handleTopicChange = (event: SelectChangeEvent<string>) => {
    const newTopicName = event.target.value as string;
    const selectedTopicObj = topics.find(
      (topic) => topic.topic_name === newTopicName
    );
    if (selectedTopicObj) {
      dispatch(setActiveTopic(selectedTopicObj));
    }
  };

  const handleOpenDialog = () => {
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setNewTopicName("");
  };

  const handleCreateTopic = () => {
    // Create a new topic via API call.
    createTopic(newTopicName)
      .then((response) => {
        // Assume the API returns the new topic object.
        dispatch(addTopic(response));
        // Set the new topic as active.
        dispatch(setActiveTopic(response));
        handleCloseDialog();
      })
      .catch((error) => {
        console.error("Error creating topic", error);
      });
  };

  return (
    <>
      <AppBar position="static" sx={{ mb: 2 }}>
        <Toolbar>
          {topics.length > 0 && (
            <Select
              value={activeTopic?.topic_name || ""}
              onChange={handleTopicChange}
              displayEmpty
              sx={{ color: "white", mr: 2 }}
            >
              {topics.map((topic) => (
                <MenuItem key={topic.uuid} value={topic.topic_name}>
                  {topic.topic_name}
                </MenuItem>
              ))}
            </Select>
          )}

          <IconButton color="inherit" onClick={handleOpenDialog}>
            <AddIcon />
          </IconButton>

          <Typography variant="h6" sx={{ flexGrow: 1, textAlign: "end" }}>
            Buddy
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Dialog for creating a new topic */}
      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>Create New Topic</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Topic Name"
            type="text"
            fullWidth
            variant="standard"
            value={newTopicName}
            onChange={(e) => setNewTopicName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleCreateTopic} disabled={!newTopicName.trim()}>
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default Header;
