import { Card, CardContent, Grid, Typography } from "@mui/material";

export const MetricThree = (props: any) => (
    <Card {...props}>
        <CardContent>
            <Grid container spacing={3}>
                <Grid item>
                    <Typography variant="h5">Metric Name</Typography>
                    <Typography>0.xx</Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);
