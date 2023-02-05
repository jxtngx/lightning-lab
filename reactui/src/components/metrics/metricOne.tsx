import {
    Avatar,
    Box,
    Card,
    CardContent,
    Grid,
    Typography,
} from "@mui/material";

export const MetricOne = (props: any) => (
    <Card sx={{ height: "100%" }} {...props}>
        <CardContent>
            <Grid
                container
                spacing={3}
                sx={{ justifyContent: "space-between" }}
            >
                <Grid item>
                    <Typography color="textPrimary" variant="h4">
                        Metric One
                    </Typography>
                </Grid>
                <Grid item>
                    <Typography
                        color="pass"
                        sx={{
                            mr: 1,
                        }}
                        variant="h4"
                    >
                        .xx%
                    </Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);
