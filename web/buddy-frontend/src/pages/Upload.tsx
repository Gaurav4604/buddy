import React from "react";
import { Box, Typography, Button } from "@mui/material";

const Upload: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5">Uploaded Files</Typography>
      <Button variant="contained" sx={{ mt: 2 }}>
        Upload New File
      </Button>
    </Box>
  );
};

export default Upload;
