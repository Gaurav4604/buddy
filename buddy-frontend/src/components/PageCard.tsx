import React from "react";
import { Card, CardContent, Typography, Box } from "@mui/material";
import { Link } from "react-router-dom";

interface PageCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  route: string;
}

const PageCard: React.FC<PageCardProps> = ({
  title,
  description,
  icon,
  route,
}) => {
  return (
    <Card
      component={Link}
      to={route}
      sx={{
        textDecoration: "none",
        display: "flex",
        flexDirection: "column",
        border: "1px solid lightgrey",
        borderRadius: 2,
        p: 2,
        width: 250,
        height: 180,
        transition: "0.3s",
        "&:hover": { boxShadow: 3 },
      }}
    >
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
          {icon}
          <Typography variant="h6" sx={{ ml: 1 }}>
            {title}
          </Typography>
        </Box>
        <Typography variant="body2" sx={{ textAlign: "start" }}>
          {description}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default PageCard;
