import { Card, CardContent, Grid, Typography } from "@mui/material";

export const MetricTwo = (props: any) => (
    <Card className="metric-container" {...props}>
        <CardContent>
            <Grid container spacing={3} sx={{ justifyContent: "space-between" }}>
                <Grid item>
                    <Typography color="textPrimary" id="card-title">
                        Metric Name
                    </Typography>
                </Grid>
                <Grid item>
                    <Typography
                        className="metric-card-text"
                    >
                        .xx%
                    </Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);
