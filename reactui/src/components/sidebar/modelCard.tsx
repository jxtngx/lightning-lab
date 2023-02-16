import "../../App.css";
import { Card, CardContent, Grid, Typography } from "@mui/material";

export const ModelCard = (props: any) => (
    <Card {...props}>
        <CardContent>
            <Grid container spacing={3}>
                <Grid item>
                    <Typography variant="h5">Model Card</Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);
