import React, { useState, useRef, useEffect } from "react";
import {
  Box,
  Typography,
  Button,
  Dialog,
  DialogContent,
  CircularProgress,
  Stack,
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import { io, Socket } from "socket.io-client";
import { createTopicContent, getTopicContent } from "../api/topic";
import UploadCard from "../components/UploadCard";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../store";
import { setTopicContent } from "../store/topicAndContentSlice";

const Upload: React.FC = () => {
  const dispatch = useDispatch();
  // Use activeTopic from state to drive API calls.
  const activeTopic = useSelector((state: RootState) => state.tnc.activeTopic);
  const topicContent = useSelector(
    (state: RootState) => state.tnc.topicContent
  );

  const [, setSelectedFiles] = useState<FileList | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [extractionMessages, setExtractionMessages] = useState<string[]>([]);
  const [apiStatus, setApiStatus] = useState<"success" | "error" | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Ref for the hidden file input
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch topic content on component mount (if activeTopic is available)
  useEffect(() => {
    if (activeTopic) {
      getTopicContent(activeTopic.topic_name)
        .then((data) => {
          dispatch(setTopicContent(data));
        })
        .catch((error) => {
          console.error("Error fetching topic content", error);
        });
    }
  }, [activeTopic, dispatch]);

  // Initialize socket connection on mount
  useEffect(() => {
    const newSocket = io("http://localhost:5000", {
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    newSocket.on("connect", () => {
      console.log("Connected to server with socket ID:", newSocket.id);
    });

    newSocket.on("connect_error", (error) => {
      console.error("Socket connection error:", error);
    });

    setSocket(newSocket);

    return () => {
      console.log("Disconnecting socket...");
      newSocket.disconnect();
    };
  }, []);

  // Listen for extraction messages via socket
  useEffect(() => {
    if (!socket) return;

    const handleExtractionMessage = (data: { message: string }) => {
      console.log("doc_extraction event received:", data);
      setExtractionMessages((prev) => [...prev, data.message]);
    };

    socket.on("doc_extraction", handleExtractionMessage);
    return () => {
      socket.off("doc_extraction", handleExtractionMessage);
    };
  }, [socket]);

  // Close the dialog after processing is complete
  useEffect(() => {
    if (!isProcessing && apiStatus !== null) {
      const timer = setTimeout(() => {
        setDialogOpen(false);
        setApiStatus(null);
        setExtractionMessages([]);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [isProcessing, apiStatus]);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFiles(event.target.files);
      handleUpload(event.target.files);
    }
  };

  const handleUpload = async (files: FileList) => {
    if (!activeTopic) {
      console.error("No active topic selected");
      return;
    }
    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append("pdfs", file);
    });

    // Determine next chapter number from current topic content length
    const nextChapterNum = topicContent.length + 1;
    formData.append("chapter_num", nextChapterNum.toString());

    // Open dialog and initialize state
    setDialogOpen(true);
    setExtractionMessages([]);
    setApiStatus(null);
    setIsProcessing(true);

    try {
      // Use activeTopic.topic_name to post the data
      await createTopicContent(activeTopic.topic_name, formData);
      setApiStatus("success");

      // Refresh topic content after upload
      const updatedContent = await getTopicContent(activeTopic.topic_name);
      dispatch(setTopicContent(updatedContent));
    } catch (error) {
      console.error("Upload error:", error);
      setApiStatus("error");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5">Uploaded Files</Typography>
      <Button variant="contained" sx={{ mt: 2 }} onClick={handleButtonClick}>
        Upload New File
      </Button>
      <input
        type="file"
        multiple
        accept=".pdf"
        style={{ display: "none" }}
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      <Dialog open={dialogOpen}>
        <DialogContent
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            p: 4,
            minWidth: 300,
            gap: 2,
          }}
        >
          <Stack spacing={1} sx={{ width: "100%" }}>
            {extractionMessages.map((msg, index) => (
              <Typography key={index} variant="body2">
                {msg}
              </Typography>
            ))}
          </Stack>
          {isProcessing ? (
            <CircularProgress />
          ) : apiStatus === "success" ? (
            <CheckCircleIcon sx={{ fontSize: 48, color: "green" }} />
          ) : apiStatus === "error" ? (
            <ErrorOutlineIcon sx={{ fontSize: 48, color: "red" }} />
          ) : null}
        </DialogContent>
      </Dialog>
      {/* Render UploadCard components using topic content from Redux */}
      <Box sx={{ mt: 4 }}>
        {topicContent.map((chapter) => (
          <UploadCard
            key={chapter.chapter_num}
            pdfName={`Chapter ${chapter.chapter_num}`}
            description={chapter.summary}
            tags={chapter.tags}
            onDelete={() => console.log("Delete Chapter", chapter.chapter_num)}
            onInsights={() =>
              console.log("Show insights for Chapter", chapter.chapter_num)
            }
          />
        ))}
      </Box>
    </Box>
  );
};

export default Upload;
