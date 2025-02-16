import React from "react";
import {
  Card,
  CardContent,
  Avatar,
  Typography,
  Box,
  Chip,
  Button,
  Stack,
} from "@mui/material";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";

interface UploadCardProps {
  pdfName: string;
  description: string;
  tags: string[];
  onDelete?: () => void;
  onInsights?: () => void;
}

const UploadCard: React.FC<UploadCardProps> = ({
  pdfName,
  description,
  tags,
  onDelete,
  onInsights,
}) => {
  return (
    <Card sx={{ width: "100%", mb: 2 }}>
      <CardContent>
        <Stack spacing={2}>
          {/* Row 1: Avatar and PDF name */}
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Avatar sx={{ bgcolor: "red", mr: 2 }}>
              <PictureAsPdfIcon />
            </Avatar>
            <Typography variant="h6">{pdfName}</Typography>
          </Box>
          {/* Row 2: Description */}
          <Box>
            <Typography variant="body1">{description}</Typography>
          </Box>
          {/* Row 3: Fixed height scrollable tag list */}
          <Box
            sx={{
              height: 100, // fixed height
              overflowY: "auto",
              display: "flex",
              flexWrap: "wrap",
              gap: 1,
              p: 1,
              border: 1,
              borderColor: "divider",
              borderRadius: 1,
            }}
          >
            {tags.map((tag, index) => (
              <Chip key={index} label={tag} />
            ))}
          </Box>
          {/* Row 4: Action buttons */}
          <Box sx={{ display: "flex", justifyContent: "flex-end", gap: 2 }}>
            <Button variant="outlined" color="error" onClick={onDelete}>
              Delete
            </Button>
            <Button variant="contained" color="primary" onClick={onInsights}>
              Insights
            </Button>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default UploadCard;
