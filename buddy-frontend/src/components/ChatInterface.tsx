import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { RootState, AppDispatch } from "../store";
import { addMessage, setMessages } from "../store/chatSlice";
import MarkdownCell from "./MarkdownCell";
import {
  Stack,
  TextField,
  InputAdornment,
  Box,
  IconButton,
  CircularProgress,
} from "@mui/material";
import ArrowUpward from "@mui/icons-material/ArrowUpward";
import PsychologyAltIcon from "@mui/icons-material/PsychologyAlt";

const ChatInterface: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const messages = useSelector((state: RootState) => state.chat.messages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch messages from the backend (simulation)
    const fetchMessages = async () => {
      const data = [
        {
          message: `### Key point about statement 1
It highlights an important idea or concept that will be central to our discussion.`,
          role: "assistant" as const,
          references: [
            {
              file: "Introduction to React",
              chapter: "Components and Props",
              page: "45",
            },
          ],
        },
        {
          message: `I totally agree with the above point!`,
          role: "user" as const,
        },
      ];
      dispatch(setMessages(data));
    };

    fetchMessages();
  }, [dispatch]);

  const handleSend = () => {
    if (loading || !input.trim()) return;

    // Dispatch user's message
    dispatch(
      addMessage({
        message: input,
        role: "user",
      })
    );

    setLoading(true);

    // Simulate a 3-second delay for the server response
    setTimeout(() => {
      dispatch(
        addMessage({
          message: `#### Key point about statement 1
It highlights an important idea or concept that will be central to our discussion.`,
          role: "assistant",
        })
      );
      setInput("");
      setLoading(false);
    }, 3000);
  };

  return (
    <Stack
      spacing={2}
      sx={{
        height: "92vh",
        p: 2,
        width: "100%",
        maxWidth: "1000px",
        justifySelf: "center",
      }}
    >
      {/* Message List */}
      <Stack spacing={1} sx={{ flex: 1, overflowY: "auto" }}>
        {messages.map((msg, index) => (
          <Box
            key={index}
            sx={{
              display: "flex",
              justifyContent:
                msg.role === "assistant" ? "flex-start" : "flex-end",
              width: "100%",
            }}
          >
            <MarkdownCell content={msg.message} references={msg.references} />
          </Box>
        ))}
      </Stack>
      {/* Message Input */}
      <TextField
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your message..."
        multiline
        minRows={1}
        maxRows={10}
        slotProps={{
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <PsychologyAltIcon />
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={handleSend}
                  sx={{ borderRadius: "50%" }}
                  disabled={loading || !input.trim()}
                >
                  {loading ? <CircularProgress size={24} /> : <ArrowUpward />}
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
        fullWidth
      />
    </Stack>
  );
};

export default ChatInterface;
