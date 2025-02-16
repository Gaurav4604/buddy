import React from "react";
import { Box, Stack, Typography } from "@mui/material";
import PageCard from "../components/PageCard";
import ChatIcon from "@mui/icons-material/Chat";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import InsightsIcon from "@mui/icons-material/BarChart";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";

const Home: React.FC = () => {
  return (
    <Stack
      sx={{ textAlign: "center", mt: 4 }}
      alignItems={"center"}
      justifyContent={"center"}
    >
      <Typography variant="h4" gutterBottom>
        Welcome to Buddy!
      </Typography>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(2, 1fr)", // 2 cards per row
          gap: 1,
          justifyContent: "center",
          alignItems: "center",
          mt: 3,
          width: "100vw",
          maxWidth: "1000px",
          "& > :nth-of-type(odd)": {
            justifySelf: "end", // Align first column to end
          },
          "& > :nth-of-type(even)": {
            justifySelf: "start", // Align second column to start
          },
        }}
      >
        <PageCard
          title="Chat"
          description="Start a conversation with Buddy, Query your Docs about anything!"
          icon={<ChatIcon />}
          route="/question"
        />
        <PageCard
          title="Uploads"
          description="Manage and view your uploaded files. View short informational summaries."
          icon={<UploadFileIcon />}
          route="/upload"
        />
        <PageCard
          title="Insights"
          description="Analyze data with graphs and charts, find Insights on your data"
          icon={<InsightsIcon />}
          route="/insights"
        />
        <PageCard
          title="Profile"
          description="View and edit your profile settings. Personalize how your data is delivered to you"
          icon={<AccountCircleIcon />}
          route="/profile"
        />
      </Box>
    </Stack>
  );
};

export default Home;
