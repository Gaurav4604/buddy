import React from "react";
import ReactMarkdown from "react-markdown";
import { Reference } from "../store/chatSlice";
import { Box, Typography, List, ListItem } from "@mui/material";

interface MarkdownCellProps {
  content: string;
}

interface MarkdownCellProps {
  content: string;
  references?: Reference[];
}

const MarkdownCell: React.FC<MarkdownCellProps> = ({ content, references }) => (
  <Box
    sx={{
      border: 1, // 1px solid border
      borderColor: "divider",
      borderRadius: 2, // Use theme spacing for border radius
      p: 2,
      width: 500,
      mb: 2,
      fontFamily: "Montserrat, sans-serif",
      "& *": {
        fontFamily: "Montserrat, sans-serif",
      },
    }}
  >
    <ReactMarkdown>{content}</ReactMarkdown>
    {references && references.length > 0 && (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          References:
        </Typography>
        <List>
          {references.map((ref, index) => (
            <ListItem key={index} disablePadding>
              <Typography variant="body2">
                {`File: ${ref.file}, Chapter: ${ref.chapter}, Page: ${ref.page}`}
              </Typography>
            </ListItem>
          ))}
        </List>
      </Box>
    )}
  </Box>
);

export default MarkdownCell;
